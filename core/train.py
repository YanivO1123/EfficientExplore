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
from core.utils import weight_reset, uncertainty_to_loss_weight

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


def update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result=False, step_count=None):
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

    if 'deep_sea' in config.env_name and step_count % config.test_interval == 0:
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
    if (config.uncertainty_architecture_type == 'rnd' or config.uncertainty_architecture_type == 'rnd_ube') \
            and config.use_uncertainty_architecture:
        rnd_loss = model.compute_value_rnd_uncertainty(hidden_state.detach())
    else:
        rnd_loss = torch.zeros(batch_size)

    # UBE loss
    if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
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
                    # Compute value RND loss
                    rnd_loss += model.compute_value_rnd_uncertainty(hidden_state.detach()) * mask_batch[:, step_i]

                    # Compute reward RND loss
                    action = action_batch[:, step_i]
                    rnd_loss += model.compute_reward_rnd_uncertainty(hidden_state.detach(), action) * mask_batch[:, step_i]

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
                # Compute value RND loss
                rnd_loss += model.compute_value_rnd_uncertainty(hidden_state.detach()) * mask_batch[:, step_i]

                # Compute reward RND loss
                action = action_batch[:, step_i]
                rnd_loss += model.compute_reward_rnd_uncertainty(hidden_state.detach(), action) * mask_batch[:, step_i]

            if 'ube' in config.uncertainty_architecture_type and config.use_uncertainty_architecture:
                ube_prediction = model.compute_ube_uncertainty(hidden_state.detach())
                ube_loss += config.ube_loss(ube_prediction, target_ube[:, step_i + 1]) * mask_batch[:, step_i]

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1) * mask_batch[:, step_i]
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1]) * mask_batch[:, step_i]
            value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i]) * mask_batch[:, step_i]
            # Follow MuZero, set half gradient
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
    # Let's test for shapes of loss
    if 'rnd' in config.uncertainty_architecture_type:
        loss += rnd_loss
    # Let's test for shapes of loss
    if 'ube' in config.uncertainty_architecture_type:
        loss += ube_loss * config.ube_loss_coeff
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

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)

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
        if config.periodic_ube_weight_reset and step_count % config.reset_ube_interval == 0 \
                and 'ube' in config.uncertainty_architecture_type:
            model.ube_network.apply(fn=weight_reset)

        # TODO: Periodically reset value and policy network weights

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

        if config.amp_type == 'torch_amp':
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result, step_count)
            scaler = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result, step_count)

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
          f"Use network prior: {config.use_network_prior} \n"
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