import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.model import concat_output, concat_output_value
from core.utils import prepare_observation_lst, LinearSchedule

import traceback

from core.visitation_counter import CountUncertainty


@ray.remote
class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):
        """CPU Batch Worker for reanalyzing targets, see Appendix.
        Prepare the context concerning CPU overhead
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        batch_storage: Any
            The batch storage (batch queue)
        mcts_storage: Ant
            The mcts-related contexts storage
        """
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.batch_storage = batch_storage
        self.mcts_storage = mcts_storage
        self.config = config

        self.last_model_index = -1
        self.batch_max_num = 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)

    # TODO: Complete the UBE context preparation - if it's even necessary, because I might want to rely on
    #  reward-value-context
    def _prepare_ube_context(self, indices, games, state_index_lst, total_transitions):
        return NotImplementedError

    def _prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        """
        prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        zero_obs = games[0].zero_obs()
        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []

        td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, config.td_steps).astype(np.int)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            rewards_lst.append(game.rewards)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        value_obs_lst = ray.put(value_obs_lst)
        reward_value_context = [value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst]
        return reward_value_context

    def _prepare_reward_value_context_w_max_targets(self, indices, games, state_index_lst, total_transitions):
        """
        prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        zero_obs = games[0].zero_obs()
        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []

        td_steps_lst = []

        # MuExplore:
        exploration_episodes = []
        zero_step_value_obs_lst = []
        zero_step_value_mask = []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, config.td_steps).astype(np.int)
            # MuExplore: store the type of episode
            exploration_episodes.append(game.exploration_episode)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            # MuExplore: Prepare the observations for bootstrapped values o_{t+0}, for max_targets
            current_value_game_obs = game.obs(state_index, config.num_unroll_steps)

            rewards_lst.append(game.rewards)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                beg_index = bootstrap_index - (state_index + td_steps)
                end_index = beg_index + config.stacked_observations

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                # MuExplore: Prepare the observations for bootstrapped values o_{t+0}, for max_targets
                if bootstrap_index - td_steps < traj_len:
                    zero_step_obs = current_value_game_obs[beg_index:end_index]
                    zero_step_value_mask.append(1)
                else:
                    zero_step_obs = zero_obs
                    zero_step_value_mask.append(0)

                value_obs_lst.append(obs)
                # MuExplore: Prepare the observations for bootstrapped values o_{t+0}, for max_targets
                zero_step_value_obs_lst.append(zero_step_obs)

        assert np.shape(value_obs_lst) == np.shape(zero_step_value_obs_lst), f"Error in computing observations for max targets. np.shape(value_obs_lst) = {np.shape(value_obs_lst)}, np.shape(zero_step_value_obs_lst) = {np.shape(zero_step_value_obs_lst)}"
        value_obs_lst = ray.put(value_obs_lst)
        # MuExplore: Prepare the observations for bootstrapped values o_{t+0}, for max_targets
        zero_step_value_obs_lst = ray.put(zero_step_value_obs_lst)

        reward_value_context = [value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, exploration_episodes, zero_step_value_obs_lst, zero_step_value_mask]
        return reward_value_context

    def _prepare_policy_non_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for non-reanalyzing part, just return the policy in self-play
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        child_visits = []
        traj_lens = []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            child_visits.append(game.child_visits)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens]
        return policy_non_re_context

    def _prepare_policy_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        zero_obs = games[0].zero_obs()
        config = self.config

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.rewards)
                child_visits.append(game.child_visits)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.stacked_observations
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_lst.append(obs)

        policy_obs_lst = ray.put(policy_obs_lst)
        policy_re_context = [policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens]
        return policy_re_context

    def make_batch(self, batch_context, ratio, weights=None):
        """prepare the context of a batch
        reward_value_context:        the context of reanalyzed value targets
        policy_re_context:           the context of reanalyzed policy targets
        policy_non_re_context:       the context of non-reanalyzed policy targets
        inputs_batch:                the inputs of batch
        weights:                     the target model weights
        Parameters
        ----------
        batch_context: Any
            batch context from replay buffer
        ratio: float
            ratio of reanalyzed policy (value is 100% reanalyzed)
        weights: Any
            the target model weights
        """
        # obtain the batch context from replay buffer
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]

            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]

            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

            # obtain the input observations
            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        re_num = int(batch_size * ratio)
        # formalize the input observations
        obs_lst = prepare_observation_lst(obs_lst)

        # formalize the inputs of a batch
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = ray.get(self.replay_buffer.get_total_len.remote())

        # obtain the context of value targets
        if self.config.use_max_value_targets or self.config.use_max_policy_targets:
            # If uses max targets, the context contains two more things: episode type, and observations to compute 0-step values
            reward_value_context = self._prepare_reward_value_context_w_max_targets(indices_lst, game_lst, game_pos_lst,
                                                                      total_transitions)
        else:
            reward_value_context = self._prepare_reward_value_context(indices_lst, game_lst, game_pos_lst,
                                                                      total_transitions)

        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        # reanalyzed policy
        if re_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_re_context(indices_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num])
        else:
            policy_re_context = None

        # non reanalyzed policy
        if re_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_re_context(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:])
        else:
            policy_non_re_context = None

        #TODO: Compute ube context with self._prepare_ube_context - if even necessary

        #TODO: Change context structure to pass UBE context
        countext = reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, weights
        self.mcts_storage.push(countext)

    def run(self):
        # start making mcts contexts to feed the GPU batch maker
        start = False
        while True:
            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                continue

            # Based on the recommendation in issue https://github.com/YeWR/EfficientZero/issues/26
            # ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
            # trained_steps, target_weights = ray.get(ray_data_lst)
            trained_steps = ray.get(self.storage.get_counter.remote())
            target_weights = None

            beta = self.beta_schedule.value(trained_steps)
            # obtain the batch context from replay buffer
            batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta))
            # break
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            new_model_index = trained_steps // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
                target_weights = ray.get(self.storage.get_target_weights.remote())
            else:
                target_weights = None

            if self.mcts_storage.get_len() < 20:
                # Observation will be deleted if replay buffer is full. (They are stored in the ray object store)
                try:
                    self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights)
                except:
                    traceback.print_exc()
                    print('Data is deleted...')
                    time.sleep(0.1)


@ray.remote(num_gpus=0.25)#num_gpus=0.125 is the original, num_gpus=0.25 runs on my machine. num_gpus=0.5  should in principle need 5GB of ram and i have 4, but i made the batch size half
class BatchWorker_GPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):
        """GPU Batch Worker for reanalyzing targets, see Appendix.
        receive the context from CPU maker and deal with GPU overheads
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        batch_storage: Any
            The batch storage (batch queue)
        mcts_storage: Ant
            The mcts-related contexts storage
        """
        self.replay_buffer = replay_buffer
        self.config = config
        self.worker_id = worker_id

        self.model = config.get_uniform_network()
        self.model.to(config.device)
        self.model.eval()

        self.mcts_storage = mcts_storage
        self.storage = storage
        self.batch_storage = batch_storage

        self.last_model_index = 0

        if 'deep_sea' in self.config.env_name:
            self.visitation_counter = CountUncertainty(name=self.config.env_name, num_envs=self.config.p_mcts_num,
                                                       mapping_seed=self.config.seed,
                                                       fake=self.config.plan_with_fake_visit_counter)

    #TODO: Complete the target generation for UBE
    def _prepare_ube(self, ube_context):
        """
            Takes a UBE context, and returns batch of UBE targets
        """
        return NotImplementedError

    def _prepare_reward_value(self, reward_value_context):
        """prepare reward and value targets from the context of rewards and values
        """
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst = reward_value_context
        value_obs_lst = ray.get(value_obs_lst)
        device = self.config.device
        batch_size = len(value_obs_lst)

        batch_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                if self.config.image_based:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                else:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float()
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            # concat the output slices after model inference
            if self.config.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
                # TODO: Do I really want to create NOISY roots as targets?
                roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)
                roots_values = roots.get_values()
                value_lst = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)

            # get last state value
            value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_size)]) ** td_steps_lst)
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            value_index = 0
            for traj_len_non_re, reward_lst, state_index in zip(traj_lens, rewards_lst, state_index_lst):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []

                horizon_id = 0
                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        value_lst[value_index] += reward * self.config.discount ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self.config.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[current_index]  # * config.discount ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        return batch_value_prefixs, batch_values

    def _prepare_reward_value_max_targets(self, reward_value_context):
        """
            prepare reward and max-value targets from the context of rewards and values.
            The max-value target is computed as follows: max(n_step_value, 0_step_value).
            The max target is only used in exploration episodes.
            The function _prepare_reward_value_max_targets() is only compatible with reward_value_context that has been
                prepared by the function _prepare_reward_value_context_w_max_targets()
        """
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, exploration_episodes, zero_step_value_obs_lst, zero_step_value_mask = reward_value_context
        value_obs_lst = ray.get(value_obs_lst)
        zero_step_value_obs_lst = ray.get(zero_step_value_obs_lst)
        device = self.config.device
        batch_size = len(value_obs_lst)

        batch_values, batch_value_prefixs, batch_value_target_types = [], [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            zero_step_value_obs_lst = prepare_observation_lst(zero_step_value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            network_output_zero_step = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                if self.config.image_based:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                    m_obs_zero_step = torch.from_numpy(zero_step_value_obs_lst[beg_index:end_index]).to(
                        device).float() / 255.0
                else:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float()
                    m_obs_zero_step = torch.from_numpy(zero_step_value_obs_lst[beg_index:end_index]).to(
                        device).float()

                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                        m_zero_step = self.model.initial_inference(m_obs_zero_step)
                else:
                    m_output = self.model.initial_inference(m_obs)
                    m_zero_step = self.model.initial_inference(m_obs_zero_step)
                network_output.append(m_output)
                network_output_zero_step.append(m_zero_step)

            # concat the output slices after model inference
            if self.config.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
                # TODO: Do I really want to create NOISY roots as targets?
                roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)
                roots_values = roots.get_values()
                value_lst = np.array(roots_values)

                # MuExplore: Do the same for zero_step_value:
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(
                    network_output_zero_step)
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [
                    np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(
                        np.float32).tolist() for _ in range(batch_size)]
                # TODO: Do I really want to create NOISY roots as targets?
                roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)
                roots_values = roots.get_values()
                value_lst_zero_step = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)
                value_lst_zero_step = concat_output_value(network_output_zero_step)

            # get last state value
            value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_size)]) ** td_steps_lst)
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            # MuExplore: set the 0-step values to the same format
            value_lst_zero_step = value_lst_zero_step.reshape(-1)   # This is a 0-step target, there's no need to discount
            value_lst_zero_step = value_lst_zero_step * np.array(zero_step_value_mask)
            value_lst_zero_step = value_lst_zero_step.tolist()

            value_index = 0
            for traj_len_non_re, reward_lst, state_index, exploration_episode in zip(traj_lens, rewards_lst, state_index_lst, exploration_episodes):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []
                # Logs whether this target is n-step EXPLORATORY target (1) or any other target - 0-step exploratory or n-step regular (0)
                targets_values_types = []

                horizon_id = 0
                value_prefix = 0.0
                base_index = state_index

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        value_lst[value_index] += reward * self.config.discount ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self.config.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        # MuExplore: Max targets:
                        # If the 0-step target > n-step target, AND we want to use max_value_targets
                        # and not only max_policy_targets, append the 0-step target
                        if value_lst[value_index] < value_lst_zero_step[value_index] and exploration_episode and \
                                self.config.use_max_value_targets:
                            target_values.append(value_lst_zero_step[value_index])
                        else:
                            target_values.append(value_lst[value_index])
                        # If this was an n-step EXPLORATORY target, append 1
                        if value_lst[value_index] > value_lst_zero_step[value_index] and exploration_episode:
                            targets_values_types.append(1)  # this was an n-step EXPLORATORY target
                        else:
                            targets_values_types.append(0)  # this was any other kind of target
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[current_index]  # * config.discount ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                        targets_values_types.append(0)
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)
                batch_value_target_types.append(targets_values_types)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        batch_value_target_types = np.asarray(batch_value_target_types)
        return batch_value_prefixs, batch_values, batch_value_target_types

    def _prepare_policy_re_max_targets(self, policy_re_context, max_targets):
        """
            prepare policy targets from the reanalyzed context of policies for max-policy targets
            Max policy targets are computed as follows:
            If exploration episode, and value target is an n-step target, DO NOT reanalyze
        """
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens = policy_re_context
        policy_obs_lst = ray.get(policy_obs_lst)
        batch_size = len(policy_obs_lst)
        device = self.config.device

        with torch.no_grad():
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                if self.config.image_based:
                    m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                else:
                    m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float()

                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()
            policy_index = 0

            for enumerate_index, (state_index, game_idx) in enumerate(zip(state_index_lst, indices)):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        # game.store_search_stats(distributions, value, current_index)
                        # MuExplore: If this was an n-step target AND an exploration episode, DO NOT use the reanlyzed policy target
                        if max_targets[enumerate_index][current_index - state_index] == 1:
                            target_policies.append(child_visits[enumerate_index][current_index])
                        else:
                            sum_visits = sum(distributions)
                            policy = [visit_count / sum_visits for visit_count in distributions]
                            target_policies.append(policy)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.asarray(batch_policies_re)
        return batch_policies_re

    def _prepare_policy_re(self, policy_re_context):
        """prepare policy targets from the reanalyzed context of policies
        """
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens = policy_re_context
        policy_obs_lst = ray.get(policy_obs_lst)
        batch_size = len(policy_obs_lst)
        device = self.config.device

        with torch.no_grad():
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                if self.config.image_based:
                    m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                else:
                    m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float()

                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()
            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        # game.store_search_stats(distributions, value, current_index)
                        sum_visits = sum(distributions)
                        policy = [visit_count / sum_visits for visit_count in distributions]
                        target_policies.append(policy)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.asarray(batch_policies_re)
        return batch_policies_re

    def _prepare_policy_non_re(self, policy_non_re_context):
        """prepare policy targets from the non-reanalyzed context of policies
        """
        batch_policies_non_re = []
        if policy_non_re_context is None:
            return batch_policies_non_re

        state_index_lst, child_visits, traj_lens = policy_non_re_context
        with torch.no_grad():
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    if current_index < traj_len:
                        target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                    else:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                        policy_mask.append(0)

                batch_policies_non_re.append(target_policies)
        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re

    def _prepare_target_gpu(self):
        input_countext = self.mcts_storage.pop()
        if input_countext is None:
            time.sleep(1)
        else:
            reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, target_weights = input_countext
            if target_weights is not None:
                self.model.load_state_dict(target_weights)
                self.model.to(self.config.device)
                self.model.eval()

            # target reward, value
            if self.config.use_max_value_targets or self.config.use_max_policy_targets:
                batch_value_prefixs, batch_values, max_targets = self._prepare_reward_value_max_targets(reward_value_context)
            else:
                batch_value_prefixs, batch_values = self._prepare_reward_value(reward_value_context)

            # target policy
            #MuExplore: If max policy targets, compute whether to reanalyze or not
            if self.config.use_max_policy_targets:
                # Identify which part of the batch is marked for reanalyses and which not
                re_num = int(self.config.batch_size * self.config.revisit_policy_search_rate)
                batch_policies_re = self._prepare_policy_re_max_targets(policy_re_context, max_targets[:re_num])
            else:
                batch_policies_re = self._prepare_policy_re(policy_re_context)

            # batch_policies_re = self._prepare_policy_re(policy_re_context)
            batch_policies_non_re = self._prepare_policy_non_re(policy_non_re_context)
            batch_policies = np.concatenate([batch_policies_re, batch_policies_non_re])

            targets_batch = [batch_value_prefixs, batch_values, batch_policies]

            # Go over this batch. look for targets that are diagonal, exploratory, and 0.0
            step_count = ray.get(self.storage.get_counter.remote())
            if self.config.use_max_value_targets and 'deep_sea/0' in self.config.env_name and step_count % self.config.test_interval == 0:
                try:
                    self.debug_deep_sea(batch_values, batch_policies, reward_value_context, max_targets, step_count)
                except:
                    traceback.print_exc()

            # a batch contains the inputs and the targets; inputs is prepared in CPU workers
            self.batch_storage.push([inputs_batch, targets_batch])

    def run(self):
        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(0.1)
                continue

            trained_steps = ray.get(self.storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break
            try:
                self._prepare_target_gpu()
            except:
                traceback.print_exc()

    def debug_deep_sea(self, batch_values, batch_policies, reward_value_context, max_targets, step_count):
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, exploration_episodes, zero_step_value_obs_lst, zero_step_value_mask = reward_value_context
        zero_step_value_obs_lst = ray.get(zero_step_value_obs_lst)

        count = 0
        exploratory_zero_targets = 0
        individual_counts = np.zeros(10)
        zero_target_exploratory_states_set = set()
        # print(f"np.shape(max_targets) = {np.shape(max_targets)} \n"
        #       f"np.shape(batch_values) = {np.shape(batch_values)} \n"
        #       f"np.shape(batch_policies) = {np.shape(batch_policies)} \n"
        #       f"np.shape(exploration_episodes) = {np.shape(exploration_episodes)} \n"
        #       f"np.shape(zero_step_value_obs_lst) = {np.shape(zero_step_value_obs_lst)} \n"
        #       f"np.shape(value_mask) = {np.shape(value_mask)} \n"
        #       f"np.shape(state_index_lst) = {np.shape(state_index_lst)} \n"
        #       )
        # For all episodes in batch
        for index_batch, exploratory_episode in enumerate(exploration_episodes):
            # For all states that are diagonal
            for index_rollout, batch_value in enumerate(batch_values[index_batch]):
                # The observations is flattened like batch_size * (rollout + 1) = 64 * 6 = 384
                # for final shape (384, 4, 10, 10).
                index_observation = index_batch * (self.config.num_unroll_steps + 1) + index_rollout
                observation = zero_step_value_obs_lst[index_observation][-1]
                row, column = self.visitation_counter.from_one_hot_state_to_indexes(observation)
                if row == column:
                    # For all values that are zero
                    if batch_values[index_batch][index_rollout] == 0:
                        if np.array_equal(np.asarray(batch_policies[index_batch][index_rollout]), np.asarray([0.5, 0.5])):
                            count += 1
                            # print(f"In reanalyze, in debug_deep_sea. State: {(row, column)}, in game: {index_batch}, "
                            #       f"exploration episode: {exploratory_episode} \n"
                            #       f"Value target = {batch_values[index_batch][index_rollout]}, "
                            #       f"policy target = {batch_policies[index_batch][index_rollout]}, "
                            #       f"value_mask = {value_mask[index_observation]} \n"
                            #       f"Used n-step target: {max_targets[index_batch, index_rollout]} \n")
                            if exploratory_episode:
                                exploratory_zero_targets += 1
                                individual_counts[row] += 1
                                zero_target_exploratory_states_set.add(state_index_lst[index_batch] // 10)
        # later_transitions = [(position // 10, position % 10)
        #                      for _, position in enumerate(zero_target_exploratory_states_set) if position > 200]
        print(f"#######################################################################\n"
              f"In reanalyze, in debug_deep_sea. Learning step = {step_count} \n "
              f"Total number of targets that are diagonal, valued 0 and policy flat: {count} \n"
              f"Out of them are from exploratory episodes: {exploratory_zero_targets}. \n"
              f"Individual count per state: {individual_counts} \n"
              f"Out of total targets of: {self.config.batch_size * (self.config.num_unroll_steps + 1)} \n"
              f"And all the individual games these targets came from, after the first 20 games: \n "
              f"{zero_target_exploratory_states_set} \n"
              f"#######################################################################\n"
              )
