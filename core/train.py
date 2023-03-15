import os
import ray
import time
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from core.log import _log
from core.test import _test
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage, QueueStorage
from core.selfplay_worker import DataWorker
from core.reanalyze_worker import BatchWorker_GPU, BatchWorker_CPU
from core.utils import weight_reset, uncertainty_to_loss_weight, prepare_observation_lst

from core.visitation_counter import CountUncertainty
import traceback


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count):
    # adjust learning rate, step lr every lr_decay_steps
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False, step_count=None,
                   rnd_optimizer=None, rnd_scaler=None):
    """update models given a batch data
    Parameters
    ----------
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    replay_buffer: Any
        replay buffer
    scaler: Any
        scaler for torch amp
    vis_result: bool
        True -> log some visualization data in tensorboard (some distributions, values, etc)
    """
    inputs_batch, targets_batch = batch
    obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
    # if config.loss_uncertainty_weighting:
    #     target_value_prefix, target_value, target_policy, target_value_uncertainties = targets_batch
    #     value_loss_weights = uncertainty_to_loss_weight(target_value_uncertainties)
    # else:
    #     target_value_prefix, target_value, target_policy = targets_batch

    if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
        target_value_prefix, target_value, target_policy, target_ube = targets_batch
        target_ube = torch.from_numpy(target_ube).to(config.device).float()
    else:
        target_value_prefix, target_value, target_policy = targets_batch
        target_ube = None

    if 'deep_sea' in config.env_name and step_count % config.test_interval == 0 and False:
        try:
            debug_train_deep_sea(obs_batch_ori, target_value, target_policy, target_value_prefix, mask_batch,
                                 config.seed, config.stacked_observations,
                                 config.batch_size, config.num_unroll_steps, step_count, action_batch,
                                 target_ube, env_name=config.env_name)
        except:
            traceback.print_exc()

    # [:, 0: config.stacked_observations * 3,:,:]
    # obs_batch_ori is the original observations in a batch
    # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
    # obs_target_batch is the observations for s_t (hidden states from representation function)
    # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames
    if config.image_based:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float() / 255.0
    else:
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float()
    obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
    obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]

    # do augmentations
    if config.use_augmentation:
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(config.device).float()
    target_value_prefix = torch.from_numpy(target_value_prefix).to(config.device).float()
    target_value = torch.from_numpy(target_value).to(config.device).float()
    target_policy = torch.from_numpy(target_policy).to(config.device).float()
    weights = torch.from_numpy(weights_lst).to(config.device).float()
    # Targets UBE already from_numpy-ed above

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_value_prefix.size(0)
    metric_loss = torch.nn.L1Loss()

    # some logs preparation
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1

    # transform targets to categorical representation
    transformed_target_value_prefix = config.scalar_transform(target_value_prefix)
    target_value_prefix_phi = config.reward_phi(transformed_target_value_prefix)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    if config.amp_type == 'torch_amp':
        with autocast():
            value, _, policy_logits, hidden_state, reward_hidden, value_variance, _ = model.initial_inference(obs_batch)
    else:
        value, _, policy_logits, hidden_state, reward_hidden, value_variance, _ = model.initial_inference(obs_batch)
    scaled_value = config.inverse_value_transform(value)

    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    predicted_value_prefixs = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits, dim=1).detach().cpu()

    # calculate the new priorities for each transition
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps

    # loss of the first step
    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    value_prefix_loss = torch.zeros(batch_size, device=config.device)
    consistency_loss = torch.zeros(batch_size, device=config.device)

    # value RND loss:
    if 'rnd' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
        rnd_loss = get_rnd_loss(model, hidden_state, batch_size, config.device, config.amp_type)
        previous_state = hidden_state
    else:
        rnd_loss = torch.zeros(batch_size)

    # UBE loss
    if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
        if config.amp_type == 'torch_amp':
            with autocast():
                ube_prediction = model.compute_ube_uncertainty(hidden_state.detach())
                ube_loss = config.ube_loss(ube_prediction, target_ube[:, 0])
        else:
            ube_prediction = model.compute_ube_uncertainty(hidden_state.detach())
            ube_loss = config.ube_loss(ube_prediction, target_ube[:, 0])
    else:
        ube_loss = torch.zeros(batch_size)

    target_value_prefix_cpu = target_value_prefix.detach().cpu()
    gradient_scale = 1 / config.num_unroll_steps
    # loss of the unrolled steps
    if config.amp_type == 'torch_amp':
        # use torch amp
        with autocast():
            for step_i in range(config.num_unroll_steps):
                # unroll with the dynamics function
                value, value_prefix, policy_logits, hidden_state, reward_hidden, _, _ = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])

                beg_index = config.image_channel * step_i
                end_index = config.image_channel * (step_i + config.stacked_observations)

                # consistency loss
                if config.consistency_coeff > 0:
                    # obtain the oracle hidden states from representation function
                    _, _, _, presentation_state, _, _, _ = model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                    # no grad for the presentation_state branch
                    dynamic_proj = model.project(hidden_state, with_grad=True)
                    observation_proj = model.project(presentation_state, with_grad=False)
                    temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                    other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                    consistency_loss += temp_loss

                if 'rnd' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
                    action = action_batch[:, step_i]
                    # Compute value RND loss
                    rnd_loss += get_rnd_loss(model, hidden_state, batch_size, config.device, config.amp_type,
                                             previous_state, action) * mask_batch[:, step_i]
                    previous_state = hidden_state

                if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
                    ube_prediction = model.compute_ube_uncertainty(hidden_state.detach())
                    ube_loss += config.ube_loss(ube_prediction, target_ube[:, step_i + 1]) * mask_batch[:, step_i]

                policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
                value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
                value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
                # Follow MuZero, set half gradient
                if config.learned_model:
                    hidden_state.register_hook(lambda grad: grad * 0.5)

                # reset hidden states
                if (step_i + 1) % config.lstm_horizon_len == 0:
                    reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                     torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

                if vis_result:
                    scaled_value_prefixs = config.inverse_reward_transform(value_prefix).detach()
                    scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                    predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                    predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                    predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                    state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                    key = 'unroll_' + str(step_i + 1) + '_l1'

                    value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                    value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                    value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                    target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                    other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                    if value_prefix_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                    if value_prefix_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                    if value_prefix_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    else:
        for step_i in range(config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden, _, _ = model.recurrent_inference(hidden_state, reward_hidden, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            # consistency loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _, _, _ = model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                # no grad for the presentation_state branch
                dynamic_proj = model.project(hidden_state, with_grad=True)
                observation_proj = model.project(presentation_state, with_grad=False)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            if 'rnd' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
                action = action_batch[:, step_i]
                # Compute value RND loss
                rnd_loss += get_rnd_loss(model, hidden_state, batch_size, config.device, config.amp_type,
                                         previous_state, action) * mask_batch[:, step_i]
                previous_state = hidden_state

            if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
                ube_prediction = model.compute_ube_uncertainty(hidden_state.detach())
                ube_loss += config.ube_loss(ube_prediction, target_ube[:, step_i + 1]) * mask_batch[:, step_i]

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
            value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
            # Follow MuZero, set half gradient
            if model.learned_model:
                hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % config.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device),
                                 torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(config.device))

            if vis_result:
                scaled_value_prefixs = config.inverse_reward_transform(value_prefix).detach()
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])
    # ----------------------------------------------------------------------------------
    # weighted loss with masks (some invalid states which are out of trajectory.)
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
            config.value_loss_coeff * value_loss + config.reward_loss_coeff * value_prefix_loss)
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(
            f"$$$$$$$$$$$$$$$$$$$$$\n There are nans / infs in original loss. loss = {loss} \n $$$$$$$$$$$$$$$$$$$$$\n")
    if torch.isnan(rnd_loss).any() or torch.isinf(rnd_loss).any():
        print(
            f"$$$$$$$$$$$$$$$$$$$$$\n There are nans / infs in rnd_loss. rnd_loss = {rnd_loss} \n $$$$$$$$$$$$$$$$$$$$$\n")
    if torch.isnan(ube_loss).any() or torch.isinf(ube_loss).any():
        print(
            f"$$$$$$$$$$$$$$$$$$$$$\n There are nans in ube_loss. ube_loss = {ube_loss} \n $$$$$$$$$$$$$$$$$$$$$\n")

    if rnd_optimizer is not None:
        rnd_optimizer.zero_grad()
        weighted_rnd_loss = (weights * (ube_loss * config.ube_loss_coeff + rnd_loss)).mean()
        if config.amp_type == 'none':
            weighted_rnd_loss.backward()
            rnd_optimizer.step()
        elif config.amp_type == 'torch_amp':
            rnd_scaler.scale(weighted_rnd_loss).backward()
            rnd_scaler.unscale_(rnd_optimizer)
            torch.nn.utils.clip_grad_norm_(model.rnd_ube_parameters(), config.max_grad_norm)
            rnd_scaler.step(rnd_optimizer)
            rnd_scaler.update()

    # Let's test for shapes of loss
    weighted_loss = (weights * loss).mean()

    if torch.isnan(weighted_loss).any():
        print(f"$$$$$$$$$$$$$$$$$$$$$\n There are nans in weighted_loss. weighted_loss = {weighted_loss} \n $$$$$$$$$$$$$$$$$$$$$\n")

    # backward
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    new_priority = value_priority
    replay_buffer.update_priorities.remote(indices, new_priority, make_time)

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean())
    if vis_result:
        reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.get_params_mean()
        other_dist['reward_weights_dist'] = reward_w_dist
        other_log['representation_weight'] = representation_mean
        other_log['dynamic_weight'] = dynamic_mean
        other_log['reward_weight'] = reward_mean

        # reward l1 loss
        value_prefix_indices_0 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        value_prefix_indices_n1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        value_prefix_indices_1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_value_prefix_base = target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
        predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
        if value_prefix_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1])
        if value_prefix_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1])
        if value_prefix_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0])

        td_data = (new_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                   target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = None, None

    return loss_data, td_data, priority_data, scaler


def _train(model, target_model, replay_buffer, shared_storage, batch_storage, config, summary_writer):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    if 'rnd_ube' in config.uncertainty_architecture_type:
        rnd_ube_params = [
            {'params': model.rnd_parameters(), 'lr': 1E-02},
            {'params': model.ube_network.parameters(), 'lr': 1E-03},
        ]
        rnd_optimizer = optim.Adam(rnd_ube_params)
        rnd_scaler = GradScaler()
        optimizer = optim.SGD(model.other_parameters(), lr=config.lr_init, momentum=config.momentum,
                              weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                              weight_decay=config.weight_decay)
        rnd_optimizer = None

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_transforms()

    print("Started collecting data to start training", flush=True)

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_transitions):
        time.sleep(1)
        pass
    print('Begin training...', flush=True)
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = model.get_weights()

    # To debug uncertainty
    if 'deep_sea' in config.env_name:
        try:
            visit_counter = CountUncertainty(name=config.env_name, num_envs=config.p_mcts_num,
                                             mapping_seed=config.seed,
                                             fake=config.plan_with_fake_visit_counter,
                                             randomize_actions=config.deepsea_randomize_actions)
        except:
            traceback.print_exc()

    # while loop
    while step_count < config.training_steps + config.last_steps:
        # remove data if the replay buffer is full. (more data settings)
        if step_count % 1000 == 0:
            replay_buffer.remove_to_fit.remote()

        # obtain a batch
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.3)
            continue
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        # Periodically reset ube weights
        if config.periodic_ube_weight_reset and step_count % config.reset_ube_interval == 0:
            reset_weights(model, config, step_count)

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        if step_count % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        # To debug uncertainty
        if 'deep_sea' in config.env_name and True:
            try:
                debug_uncertainty(model=model, config=config, training_step=step_count, device=config.device,
                                  visit_counter=visit_counter, batch=batch)
            except:
                traceback.print_exc()
            model.train()

        if config.amp_type == 'torch_amp':
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result, step_count,
                                      rnd_optimizer, rnd_scaler)
            scaler = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result, step_count,
                                      rnd_optimizer, rnd_scaler)

        if step_count % config.log_interval == 0:
            _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage, summary_writer, vis_result)

        # The queue is empty.
        if step_count >= 100 and step_count % 50 == 0 and batch_storage.get_len() == 0:
            print('Warning: Batch Queue is empty (Require more batch actors Or batch actor fails).')

        step_count += 1

        # save models
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(model.get_weights())
    time.sleep(30)
    return model.get_weights()


def train(config, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    print(f"MuExplore hyperparameters configuration: \n"
          f"1. Exploration parameters: \n"
          f"Using deep exploration: {config.use_deep_exploration} \n"
          f"Using MuExplore - propagating uncertainty in MCTS: {config.mu_explore} \n"
          f"Deep exploration is based on UBE without MuExplore: "
          f"{not config.mu_explore and 'ube' in config.uncertainty_architecture_type} \n"
          f"Number of exploratory environments: {config.number_of_exploratory_envs} out of {config.p_mcts_num} envs total \n"
          f"Beta value = {config.beta} \n"
          f"Disable policy usage in exploration: {config.disable_policy_in_exploration} \n"
          f"Using visitation counter: {config.use_visitation_counter} \n"
          f"Planning with visitation counter: {config.plan_with_visitation_counter} \n"
          f"Using state-visits (True), or state-action visits (False): {config.plan_with_state_visits} \n"
          f"Using FAKE visitation counter: {config.plan_with_fake_visit_counter} \n"
          f"\n"
          
          f"2. Uncertainty-architecture parameters: \n"
          f"Using uncertainty architecture: {config.use_uncertainty_architecture} \n"
          f"Type of uncertainty architecture: {config.uncertainty_architecture_type} \n"
          f"Ensemble size: {config.ensemble_size} \n"
          f"Use network prior: {config.use_prior} \n"
          f"\n"
          
          f"3. Exploration-targets parameters: \n"
          f"use_max_value_targets = {config.use_max_value_targets} \n"
          f"use_max_policy_targets = {config.use_max_policy_targets} \n"
          f"Using learned model (MuZero): {config.learned_model}, or given dynamics model (Mu-AlphaZero): {not config.learned_model} \n"
          f"\n"
          
          f"Starting workers"
          , flush=True)

    storage = SharedStorage.remote(model, target_model)

    # prepare the batch and mctc context storage
    batch_storage = QueueStorage(15, 20)
    mcts_storage = QueueStorage(18, 25)
    replay_buffer = ReplayBuffer.remote(config=config)

    # other workers
    workers = []

    # reanalyze workers
    print("Starting Reanalyze workers", flush=True)
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    gpu_workers = [BatchWorker_GPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.gpu_actor)]
    workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]

    # self-play workers
    print("Starting data_workers", flush=True)
    data_workers = [DataWorker.remote(rank, replay_buffer, storage, config) for rank in range(0, config.num_actors)]
    workers += [worker.run.remote() for worker in data_workers]
    # test workers
    print("Starting test workers", flush=True)
    workers += [_test.remote(config, storage)]

    # training loop
    print("Calling _train", flush=True)
    final_weights = _train(model, target_model, replay_buffer, storage, batch_storage, config, summary_writer)

    ray.wait(workers)
    print('Training over...', flush=True)

    return model, final_weights

def debug_train_deep_sea(observations_batch, values_targets_batch, policy_targets_batch, rewards_targets_batch,
                         mask_batch, action_mapping_seed, stacked_observations, batch_size, rollout_length,
                         step_count, action_batch, ube_targets_batch, env_name):
    # Process the observations if necessary
    # observations_batch is of shape (batch_size, stacked_observations * unroll_size, h, w) [:, 0: config.stacked_observations * 3,:,:]
    # observations_batch = observations_batch
    visitation_counter = CountUncertainty(name=env_name, num_envs=1, mapping_seed=action_mapping_seed)
    # Search the batch for all states
    diagonal_observation_indexes = []
    diagonal_observations = []
    # print(f"=========================================================================== \n"
    #       f"np.shape(observations_batch) = {np.shape(observations_batch)} and expecting (batch_size, stack_obs + unroll = 4 + 5 = 9, 10, 10) \n"
    #       f"np.shape(values_targets_batch) = {np.shape(values_targets_batch)} \n"
    #       f"np.shape(policy_targets_batch) = {np.shape(policy_targets_batch)} \n"
    #       f"np.shape(mask_batch) = {np.shape(mask_batch)} \n"
    #       f"=========================================================================== \n"
    #       )
    zero_obs = np.zeros(shape=(10, 10))
    for i in range(batch_size):
        observation_rollout = observations_batch[i, :, :, :]
        for j in range(rollout_length + 1):
            observation = observation_rollout[j + stacked_observations - 1, :, :]
            if not np.array_equal(observation, zero_obs):
                row, column = visitation_counter.from_one_hot_state_to_indexes(observation)
                # Identify the indexes all the states that are on the diagonal
                if row == column:
                    diagonal_observation_indexes.append([i, j])
                    diagonal_observations.append([row, column])
            elif np.array_equal(observation, zero_obs):
                # If this observation is part of the previous trajectory, AND this trajectory is along a diagonal
                # AND it's the next observation in this trajectory
                if len(diagonal_observation_indexes) > 0 and i == diagonal_observation_indexes[-1][0] and j == diagonal_observation_indexes[-1][1] + 1:
                    diagonal_observation_indexes.append([i, j])
                    diagonal_observations.append([-1, -1])

    print(f"step_count = {step_count}, Num targets total = {batch_size * (rollout_length + 1)}, ouf of are diagonal = {len(diagonal_observation_indexes)}")
    for index, state in zip(diagonal_observation_indexes, diagonal_observations):
        [i, j] = index
        print(f"Trajectory = {i}, state: {state}, "
              f"mask: {mask_batch[i, j - 1] if j > 0 else 1},"
              f" value target: {values_targets_batch[i, j]}, "
              f"reward target {rewards_targets_batch[i, j]},"
              f" policy target: {policy_targets_batch[i, j]}, "
              f"ube_target: {ube_targets_batch[i, j] if ube_targets_batch is not None else None}, "
              f"action: {action_batch[i, j - 1] if j > 0 else 1}")


def get_rnd_loss(model, next_state, batch_size, device, amp_type, previous_state=None, action=None):
    """
        To compute the rnd error for value, we take the next-state prediction for previous-state and action.
        To compute the rnd error for reward, we take the previous-state and action.
        This is the reason both previous state and next state are passed to this function.
    """
    local_loss = torch.zeros(batch_size, device=device)

    # Compute value_rnd loss
    state_for_rnd = next_state.reshape(-1, model.input_size_value_rnd).detach()
    local_loss += torch.nn.functional.mse_loss(model.value_rnd_network(state_for_rnd),
                                               model.value_rnd_target_network(state_for_rnd).detach(),
                                               reduction='none').sum(dim=-1)
    # Compute reward_rnd loss
    if action is not None and previous_state is not None:
        action_one_hot = (
            torch.zeros(
                (
                    previous_state.shape[0],  # batch dimension
                    model.action_space_size
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        flattened_state = previous_state.reshape(-1, model.input_size_reward_rnd - model.action_space_size)
        state_action = torch.cat((flattened_state, action_one_hot), dim=1).detach()
        local_loss += torch.nn.functional.mse_loss(model.reward_rnd_network(state_action),
                                                   model.reward_rnd_target_network(state_action).detach(),
                                                   reduction='none').sum(dim=-1)
    return local_loss


def reset_weights(model, config, step_count):
    if 'ube' in config.uncertainty_architecture_type:
        model.ube_network.apply(fn=weight_reset)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              f"ube_network has been reset, at training step: {step_count}\n"
              f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
              , flush=True)
    if 'rnd' in config.uncertainty_architecture_type:
        model.reward_rnd_network.apply(fn=weight_reset)
        model.reward_rnd_target_network.apply(fn=weight_reset)
        model.value_rnd_network.apply(fn=weight_reset)
        model.value_rnd_target_network.apply(fn=weight_reset)
        for p in model.value_rnd_target_network.parameters():
            with torch.no_grad():
                p *= 4
        for p in model.reward_rnd_target_network.parameters():
            with torch.no_grad():
                p *= 4
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              f"both rnd networks have been reset, at training step: {step_count} \n"
              f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
              , flush=True)


def debug_uncertainty(model, config, training_step, device, visit_counter, batch):
    model.eval()
    with torch.no_grad():
        env_size = config.env_size
        inputs_batch, targets_batch = batch
        obs_batch_ori, action_batch, _, _, _, _ = inputs_batch
        actions = torch.from_numpy(action_batch).to(config.device).unsqueeze(-1).long()
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(config.device).float()
        trained_states = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]

        target_value_prefix, target_value, target_policy, target_ube = targets_batch

        if training_step % config.reset_ube_interval == 0:
            visit_counter.s_counts = np.zeros(shape=visit_counter.observation_space_shape)
            visit_counter.sa_counts = np.zeros(shape=(visit_counter.observation_space_shape + (visit_counter.action_space,)))

        if training_step % 50 == 0:
            # Setup observations for every state in the 10 by 10 deep_sea
            num_columns = env_size
            num_rows = env_size
            batched_obs = []
            for row in range(num_rows):
                row_observations = []
                for column in range(num_columns):
                    current_obs = np.zeros(shape=(num_rows, num_columns))
                    current_obs[row, column] = 1
                    current_obs = current_obs[np.newaxis, :]
                    row_observations.append(current_obs)
                batched_obs.append(row_observations)

            # And 10 zeros states:
            row_observations = []
            for column in range(num_columns):
                current_obs = np.zeros(shape=(num_rows, num_columns))
                current_obs = current_obs[np.newaxis, :]
                row_observations.append(current_obs)
            batched_obs.append(row_observations)

            # Setup metrics we're interested in
            row_value_rnds = []
            recurrent_inf_reward_unc_prediction_left = []
            recurrent_inf_reward_unc_prediction_right = []
            final_value_uncertainty_prediction = []
            row_ube_predictions = []
            row_reward_rnds_left = []
            row_reward_rnds_right = []

            # Compute uncertainty metrics for each state, batched row by row
            for row in range(num_rows + 1):
                stack_obs = prepare_observation_lst(batched_obs[row])
                stack_obs = torch.from_numpy(stack_obs).to(device)
                if config.amp_type == 'torch_amp':
                    with autocast():
                        network_output_init = model.initial_inference(stack_obs.float())

                        # Prepare for recurrent inf
                        hidden_states = torch.from_numpy(network_output_init.hidden_state).float().to(device)
                        hidden_states_c_reward = torch.from_numpy(network_output_init.reward_hidden[0]).float().to(device)
                        hidden_states_h_reward = torch.from_numpy(network_output_init.reward_hidden[1]).float().to(device)
                        actions_left = torch.zeros(size=(num_columns, 1)).to(device).long()
                        actions_right = torch.ones(size=(num_columns, 1)).to(device).long()

                        value_rnds = model.compute_value_rnd_uncertainty(hidden_states)
                        ube_predictions = model.compute_ube_uncertainty(hidden_states)

                        network_output_recur_left = model.recurrent_inference(hidden_states,
                                                                              (hidden_states_c_reward, hidden_states_h_reward),
                                                                              actions_left)
                        network_output_recur_right = model.recurrent_inference(hidden_states,
                                                                               (hidden_states_c_reward, hidden_states_h_reward),
                                                                               actions_right)
                else:
                    network_output_init = model.initial_inference(stack_obs.float())

                    # Prepare for recurrent inf
                    # hidden_states = network_output_init.hidden_state
                    # hidden_states_c_reward = network_output_init.reward_hidden[0]
                    # hidden_states_h_reward = network_output_init.reward_hidden[1]
                    hidden_states = torch.from_numpy(network_output_init.hidden_state).float().to(device)
                    hidden_states_c_reward = torch.from_numpy(network_output_init.reward_hidden[0]).float().to(device)
                    hidden_states_h_reward = torch.from_numpy(network_output_init.reward_hidden[1]).float().to(device)
                    actions_left = torch.zeros(size=(num_columns, 1)).to(device).long()
                    actions_right = torch.ones(size=(num_columns, 1)).to(device).long()

                    value_rnds = model.compute_value_rnd_uncertainty(hidden_states)
                    ube_predictions = model.compute_ube_uncertainty(hidden_states)
                    reward_rnds_left = model.compute_reward_rnd_uncertainty(hidden_states, actions_left)
                    reward_rnds_right = model.compute_reward_rnd_uncertainty(hidden_states, actions_right)

                    network_output_recur_left = model.recurrent_inference(hidden_states,
                                                                          (hidden_states_c_reward, hidden_states_h_reward),
                                                                          actions_left)
                    network_output_recur_right = model.recurrent_inference(hidden_states,
                                                                           (hidden_states_c_reward, hidden_states_h_reward),
                                                                           actions_right)

                row_value_rnds.append(value_rnds)
                recurrent_inf_reward_unc_prediction_left.append(network_output_recur_left.value_prefix_variance)
                recurrent_inf_reward_unc_prediction_right.append(network_output_recur_right.value_prefix_variance)
                final_value_uncertainty_prediction.append(network_output_init.value_variance)
                row_ube_predictions.append(ube_predictions)
                row_reward_rnds_left.append(reward_rnds_left)
                row_reward_rnds_right.append(reward_rnds_right)

            final_value_uncertainty_prediction = np.stack(final_value_uncertainty_prediction, axis=0)
            row_value_rnds = torch.stack(row_value_rnds, dim=0)
            row_ube_predictions = torch.stack(row_ube_predictions, dim=0)
            row_reward_rnds_left = torch.stack(row_reward_rnds_left, dim=0)
            row_reward_rnds_right = torch.stack(row_reward_rnds_right, dim=0)
            recurrent_inf_reward_unc_prediction_left = np.stack(recurrent_inf_reward_unc_prediction_left, axis=0)
            recurrent_inf_reward_unc_prediction_right = np.stack(recurrent_inf_reward_unc_prediction_right, axis=0)

            # Print state counts
            print(f"*********************************************************\n"
                  f"At training step {training_step}, debug-prints for RND and UBE.")
            print(f"Printing state counts: \n"
                  f"{visit_counter.s_counts}"
                  , flush=True)
            # print(f"Value RND scores for states: \n"
            #       f"{row_value_rnds}", flush=True)
            print(f"UBE + RND scores for states: \n"
                  f"{final_value_uncertainty_prediction}", flush=True)
            # print(f"UBE scores for states: \n{row_ube_predictions}", flush=True)
            print(f"The targets for UBE at THIS SPECIFIC time step: \n{target_ube}", flush=True)
            # print(f"Now printing state-action \n "
            #       f"State-action visitations counts: \n"
            #       f"{visit_counter.sa_counts}"
            #       , flush=True)
            # print(f"Reward uncertainty for action 0: \n{row_reward_rnds_left}", flush=True)
            # print(f"Reward uncertainty for action 1: \n{row_reward_rnds_right}", flush=True)
            # print(f"recurrent_inf_reward_unc_prediction_left: \n{recurrent_inf_reward_unc_prediction_left}", flush=True)
            # print(f"recurrent_inf_reward_unc_prediction_right: \n{recurrent_inf_reward_unc_prediction_right} ", flush=True)
            # print(f"Finally, let's do one more prediction for the zeros-state")
            # stack_obs = torch.zeros(size=(2, 1, num_rows, num_columns,)).to(device).float()
            # if config.amp_type == 'torch_amp':
            #     with autocast():
            #         network_output_init = model.initial_inference(stack_obs)
            #         hidden_states = network_output_init.hidden_state
            #         value_rnds = model.compute_value_rnd_uncertainty(hidden_states)
            # else:
            #     network_output_init = model.initial_inference(stack_obs)
            #     hidden_states = network_output_init.hidden_state
            #     value_rnds = model.compute_value_rnd_uncertainty(hidden_states)
            # print(f"UBE prediction for the zero obs is: {network_output_init.value_variance[0]} \n"
            #       f"value_RND prediction for the zero obs is: {value_rnds[0]}")

        # Because this function is called BEFORE training, we observe AFTER we print
        # Observe all states in trained_states with the visit_counter
        batch_size = trained_states.shape[0]
        actions = actions.cpu().long().numpy().squeeze()
        for i in range(batch_size):
            observation = trained_states[i, -1, :, :].cpu().long().numpy()
            action = actions[i]
            visit_counter.observe(observation, action)
    model.train()
