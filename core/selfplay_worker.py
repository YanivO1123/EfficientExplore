import copy

import ray
import time
import torch
import os
import traceback
import numpy as np
import core.ctree.cytree as cytree
import math
from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, select_q_based_action

from core.visitation_counter import CountUncertainty


@ray.remote(num_gpus=0.125)
class DataWorker(object):
    def __init__(self, rank, replay_buffer, storage, config):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        # double buffering when data is sufficient
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1

        self.visitation_counter = None

    def put(self, data):
        # put a game history into the pool
        self.trajectory_pool.append(data)

    def len_pool(self):
        # current pool size
        return len(self.trajectory_pool)

    def free(self):
        # save the game histories and clear the pool
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories):
        """put the last game history into the pool if the current game is finished
        Parameters
        ----------
        last_game_histories: list
            list of the last game histories
        last_game_priorities: list
            list of the last game priorities
        game_histories: list
            list of the current game histories
        """
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        last_game_histories[i].game_over()

        self.put((last_game_histories[i], last_game_priorities[i]))
        self.free()

        # reset last block
        last_game_histories[i] = None
        last_game_priorities[i] = None

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        # obtain the priorities at index i
        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self):
        print(f"Data worker started", flush=True)
        # number of parallel mcts
        env_nums = self.config.p_mcts_num
        model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()
        print(f"Model started", flush=True)
        start_training = False
        envs = [self.config.new_game(self.config.seed + (self.rank + 1) * i) for i in range(env_nums)]
        if self.config.use_deep_exploration:
            exploit_env_nums = env_nums - self.config.number_of_exploratory_envs
        else:
            exploit_env_nums = env_nums
        print(f"Envs started", flush=True)
        # MuExplore: start the visitation counter if it's wanted
        if self.config.use_visitation_counter and 'deep_sea' in self.config.env_name:
            self.visitation_counter = CountUncertainty(name=self.config.env_name, num_envs=env_nums,
                                                       mapping_seed=self.config.seed,
                                                       fake=self.config.plan_with_fake_visit_counter,
                                                       randomize_actions=self.config.deepsea_randomize_actions)
            print(f"Initiated visitation counter", flush=True)
        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # 100k benchmark
        total_transitions = 0
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors

        # MuExplore: keep track of values and value uncertainties to debug beta
        value_max, value_unc_max, value_min, value_unc_min, value_sum, value_unc_sum, value_unc_count = -math.inf, -math.inf, math.inf, math.inf, 0, 0, 0
        # To alternate random action selection or det. action selection in deep_sea
        deterministic_deep_sea = False
        # We change action selection in deepsea from random to det. every N = 1 batched episodes
        flip_deep_sea_action_selection = 1 * self.config.env_size * self.config.p_mcts_num
        # For q_value_based action selection
        pb_c_init = self.config.pb_c_init

        try:
            with torch.no_grad():
                while True:
                    # Update the state counter in the shared storage every episode
                    if self.config.use_visitation_counter:
                        current_s_counts = copy.deepcopy(self.visitation_counter.s_counts)
                        current_sa_counts = copy.deepcopy(self.visitation_counter.sa_counts)
                        self.storage.set_counts.remote(current_s_counts, current_sa_counts)
                        np.save(self.config.exp_path + "/s_counts", np.asarray(current_s_counts))
                        np.save(self.config.exp_path + "/sa_counts", np.asarray(current_sa_counts))

                    trained_steps = ray.get(self.storage.get_counter.remote())
                    # training finished
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        time.sleep(30)
                        break

                    init_obses = [env.reset() for env in envs]
                    dones = np.array([False for _ in range(env_nums)])
                    game_histories = [GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                  config=self.config,
                                                  exploration_episode=(i >= exploit_env_nums and self.config.use_deep_exploration))
                                      for i in range(env_nums)]
                    last_game_histories = [None for _ in range(env_nums)]
                    last_game_priorities = [None for _ in range(env_nums)]

                    # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
                    stack_obs_windows = [[] for _ in range(env_nums)]

                    for i in range(env_nums):
                        stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                        game_histories[i].init(stack_obs_windows[i])

                    # for priorities in self-play
                    search_values_lst = [[] for _ in range(env_nums)]
                    pred_values_lst = [[] for _ in range(env_nums)]

                    # some logs
                    eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums), np.zeros(env_nums)
                    step_counter = 0

                    self_play_rewards = 0.
                    self_play_ori_rewards = 0.
                    self_play_moves = 0.
                    self_play_episodes = 0.

                    self_play_rewards_max = - np.inf
                    self_play_moves_max = 0

                    self_play_visit_entropy = []
                    other_dist = {}

                    # play games until max moves
                    while not dones.all() and (step_counter <= self.config.max_moves):
                        if not start_training:
                            start_training = ray.get(self.storage.get_start_signal.remote())

                        # get model
                        trained_steps = ray.get(self.storage.get_counter.remote())

                        if trained_steps >= self.config.training_steps + self.config.last_steps:
                            # training is finished
                            time.sleep(30)
                            return
                        if start_training and self.config.training_ratio * (total_transitions / max_transitions) > \
                                (trained_steps / self.config.training_steps):
                            # self-play is faster than training speed or finished
                            # MuExplore:
                            # added training_ratio in config, which requires ratio training / interactions to be AT LEAST
                            # training_ratio
                            time.sleep(1)
                            continue

                        # set temperature for distributions
                        _temperature = np.array(
                            [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps) for env in
                             envs])

                        # update the models in self-play every checkpoint_interval
                        new_model_index = trained_steps // self.config.checkpoint_interval
                        if new_model_index > self.last_model_index:
                            self.last_model_index = new_model_index
                            # update model
                            weights = ray.get(self.storage.get_weights.remote())
                            model.set_weights(weights)
                            model.to(self.device)
                            model.eval()

                            # log if more than 1 env in parallel because env will reset in this loop.
                            if env_nums > 1:
                                if len(self_play_visit_entropy) > 0:
                                    visit_entropies = np.array(self_play_visit_entropy).mean()
                                    visit_entropies /= max_visit_entropy
                                else:
                                    visit_entropies = 0.

                                if self_play_episodes > 0:
                                    log_self_play_moves = self_play_moves / self_play_episodes
                                    log_self_play_rewards = self_play_rewards / self_play_episodes
                                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                                else:
                                    log_self_play_moves = 0
                                    log_self_play_rewards = 0
                                    log_self_play_ori_rewards = 0

                                self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                                self_play_rewards_max, _temperature.mean(),
                                                                                visit_entropies, 0,
                                                                                other_dist)

                                self_play_rewards_max = - np.inf

                        step_counter += 1
                        for i in range(env_nums):
                            # reset env if finished
                            if dones[i]:

                                # pad over last block trajectory
                                if last_game_histories[i] is not None:
                                    self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                                # store current block trajectory
                                priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                                game_histories[i].game_over()

                                self.put((game_histories[i], priorities))
                                self.free()

                                # reset the finished env and new a env
                                envs[i].close()
                                init_obs = envs[i].reset()
                                game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                                config=self.config,
                                                                exploration_episode=(i >= exploit_env_nums and self.config.use_deep_exploration))
                                last_game_histories[i] = None
                                last_game_priorities[i] = None
                                stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                                game_histories[i].init(stack_obs_windows[i])

                                # log
                                self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                                self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                                self_play_rewards += eps_reward_lst[i]
                                self_play_ori_rewards += eps_ori_reward_lst[i]
                                self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                                self_play_moves += eps_steps_lst[i]
                                self_play_episodes += 1

                                pred_values_lst[i] = []
                                search_values_lst[i] = []
                                # end_tags[i] = False
                                eps_steps_lst[i] = 0
                                eps_reward_lst[i] = 0
                                eps_ori_reward_lst[i] = 0
                                visit_entropies_lst[i] = 0

                        # stack obs for model inference
                        stack_obs = [game_history.step_obs() for game_history in game_histories]
                        if self.config.use_visitation_counter and self.visitation_counter is not None:
                            # Take the last of observation of stacked obs from shape (num_envs, h, w)
                            # to shape (num_envs, 2)
                            initial_observations_for_counter = self.visitation_counter.from_one_hot_state_to_indexes(np.array(stack_obs, dtype=np.uint8)[:,-1,:,:])
                        if self.config.image_based:
                            stack_obs = prepare_observation_lst(stack_obs)
                            stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                        elif "deep_sea" in self.config.env_name:
                            stack_obs = prepare_observation_lst(stack_obs)
                            stack_obs = torch.from_numpy(stack_obs).to(self.device)
                        else:
                            stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)

                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                network_output = model.initial_inference(stack_obs.float())
                        else:
                            network_output = model.initial_inference(stack_obs.float())
                        hidden_state_roots = network_output.hidden_state
                        reward_hidden_roots = network_output.reward_hidden
                        value_prefix_pool = network_output.value_prefix
                        policy_logits_pool = network_output.policy_logits.tolist()
                        value_prefix_variance_pool = network_output.value_prefix_variance.tolist()

                        noises = [
                            np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(
                                np.float32).tolist() for _ in range(env_nums)]

                        # MuExplore:
                        if self.config.mu_explore:
                            # Disable policy prior in exploration
                            if self.config.disable_policy_in_exploration:
                                policy_logits_pool = policy_logits_pool[:exploit_env_nums] + \
                                                     [np.ones_like(policy_logits_pool[0]).tolist()
                                                      for _ in range(len(policy_logits_pool) - exploit_env_nums)]
                            # Init Exploratory CRoots and prepare them exploratorily
                            roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations,
                                                 self.config.beta, self.config.number_of_exploratory_envs)
                            roots.prepare_explore(self.config.root_exploration_fraction, noises, value_prefix_pool,
                                                  policy_logits_pool, value_prefix_variance_pool, self.config.beta,
                                                  self.config.number_of_exploratory_envs)
                        else:
                            roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations)
                            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool,
                                          policy_logits_pool)

                        # do MCTS for a policy
                        if self.config.plan_with_visitation_counter and self.visitation_counter is not None:
                            # If we plan with a visitation counter for MuExplore (only implemented for deep_sea)
                            MCTS(self.config).search_w_visitation_counter(roots, model, hidden_state_roots,
                                                                          reward_hidden_roots,
                                                                          self.visitation_counter,
                                                                          initial_observations_for_counter,
                                                                          use_state_visits=self.config.plan_with_state_visits,
                                                                          sampling_times=self.config.sampling_times)
                        else:   # Otherwise
                            MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots, acting=True)

                        roots_distributions = roots.get_distributions()
                        roots_values = roots.get_values()

                        # If UBE + q_action_selection + started training + not muexplore + yes ube + deep exploration:
                        # compute the ube q_unc prediction for each action
                        if not self.config.standard_action_selection and start_training and not self.config.mu_explore \
                                and 'ube' in self.config.uncertainty_architecture_type and self.config.use_deep_exploration:
                            # Organized as [actions, environments]
                            ube_predictions = self.get_ube_predictions(hidden_state_roots, reward_hidden_roots, env_nums, model)

                        for i in range(env_nums):
                            deterministic = False
                            if start_training:
                                distributions, value, temperature, env = roots_distributions[i], roots_values[i], _temperature[i], envs[i]
                            else:
                                # before starting training, use random policy
                                value, temperature, env = roots_values[i], _temperature[i], envs[i]
                                distributions = np.ones(self.config.action_space_size)

                            # In deep_sea, f > 50% of transitions or using counts without UBE, act det.
                            if start_training and 'deep_sea' in self.config.env_name and (
                                    total_transitions > self.config.total_transitions / 2 or (
                                             self.config.plan_with_visitation_counter
                                             and 'ube' not in self.config.uncertainty_architecture_type
                                             and self.config.mu_explore
                                     )
                                    ):
                                deterministic = True
                            elif start_training and 'deep_sea' in self.config.env_name \
                                    and total_transitions % flip_deep_sea_action_selection == 0:
                                deterministic_deep_sea = not deterministic_deep_sea
                                deterministic = deterministic_deep_sea

                            # If configured to use q_based action selection in exploration episodes
                            if not self.config.standard_action_selection and start_training and i >= exploit_env_nums:
                                q_values = roots.get_roots_children_values(self.config.discount)[i]
                                # In exploration, the Q values are: Q + Q_unc. In MuExplore, Q_unc is computed by MCTS
                                if self.config.mu_explore:
                                    q_uncertainties = roots.get_roots_children_uncertainties(self.config.discount)[i]
                                    q_values = [q_value + self.config.beta * uncertainty
                                                for (q_value, uncertainty) in zip(q_values, q_uncertainties)]
                                # With UBE and WITHOUT MuExplore, ube_predictions need to be computed by the NN.
                                elif 'ube' in self.config.uncertainty_architecture_type:
                                    q_uncertainties = ube_predictions[:, i]     # For all actions :, for environment i
                                    q_values = [q_value + self.config.beta * uncertainty
                                                for (q_value, uncertainty) in zip(q_values, q_uncertainties)]
                                else:
                                    print(f"WARNING: Using select_q_based_action in exploration episodes WITHOUT either "
                                          f"MuExplore or UBE. If this is intended, message can be ignored.")
                                action, visit_entropy = select_q_based_action(q_values, distributions, c=pb_c_init,
                                                                              temperature=temperature,
                                                                              deterministic=deterministic)
                            else:
                                action, visit_entropy = select_action(distributions, temperature=temperature,
                                                                      deterministic=deterministic)
                            # MuExplore: Add state-action to visitation counter
                            if self.config.use_visitation_counter and i >= exploit_env_nums:
                                # Take the last observation that was stored, and the current action
                                self.visitation_counter.observe(game_histories[i].obs_history[-1], action)

                            obs, ori_reward, done, info = env.step(action)

                            if ori_reward > 0 and 'deep_sea' in self.config.env_name:
                                ori_reward = ori_reward * 10
                                if self.config.use_visitation_counter and self.visitation_counter is not None:
                                    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                                          f"$$$$$$$$$$$$$$$$ \n"
                                          f"Encountered reward: {ori_reward}. Env index is :{i}. "
                                          f"Transition is {total_transitions}. "
                                          f"State is: {initial_observations_for_counter[i]}, and action is: {action} \n"
                                          f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                                          f"$$$$$$$$$$$$$$$$ \n"
                                          , flush=True)

                            try:
                                if total_transitions % self.config.test_interval == 0:
                                    if self.config.use_visitation_counter:
                                        print(
                                            f"Printing the state-action visitation counter at the last row, at transition number {total_transitions}: \n"
                                            f"{self.visitation_counter.sa_counts[-1, :, :]} \n"
                                            # f"Visitations to actions at bottom-right-corner-state: {self.visitation_counter.sa_counts[-1,-1]} \n"
                                            f"Printing the state visitation counter: \n"
                                            f"{self.visitation_counter.s_counts}"
                                            , flush=True)
                                if self.config.mu_explore and total_transitions > 0:
                                    root_values_uncertainties = roots.get_values_uncertainty()
                                    value_max = max(value_max, np.max(roots_values))
                                    value_min = min(value_min, np.min(roots_values))
                                    value_sum += sum(roots_values)
                                    value_unc_max = max(value_unc_max, np.max(root_values_uncertainties[exploit_env_nums:]))
                                    value_unc_min = min(value_unc_min, np.min(root_values_uncertainties[exploit_env_nums:]))
                                    value_unc_sum += sum(root_values_uncertainties[exploit_env_nums:])
                                    value_unc_count += len(root_values_uncertainties[exploit_env_nums:])
                                    if total_transitions % self.config.test_interval == 0:
                                        print(
                                            f"Printing root-values and root-values-uncertainties statistics at transition number "
                                            f"{total_transitions}: \n"
                                            f"values: max = {value_max}, min: {value_min}, mean: "
                                            f"{value_sum / total_transitions} \n"
                                            f"value uncertainties: max = {value_unc_max}, min = {value_unc_min}, mean = "
                                            f"{value_unc_sum / value_unc_count} \n"
                                            , flush=True)
                                        if 'deep_sea/0' in self.config.env_name:
                                            self.debug_deep_sea(model)
                            except:
                                traceback.print_exc()

                            # clip the reward
                            if self.config.clip_reward:
                                clip_reward = np.sign(ori_reward)
                            else:
                                clip_reward = ori_reward

                            # store data
                            game_histories[i].store_search_stats(distributions, value)
                            game_histories[i].append(action, obs, clip_reward)

                            eps_reward_lst[i] += clip_reward
                            eps_ori_reward_lst[i] += ori_reward
                            dones[i] = done
                            visit_entropies_lst[i] += visit_entropy

                            eps_steps_lst[i] += 1
                            total_transitions += 1

                            if self.config.use_priority and not self.config.use_max_priority and start_training:
                                pred_values_lst[i].append(network_output.value[i].item())
                                search_values_lst[i].append(roots_values[i])

                            # fresh stack windows
                            del stack_obs_windows[i][0]
                            stack_obs_windows[i].append(obs)

                            # if game history is full;
                            # we will save a game history if it is the end of the game or the next game history is finished.
                            if game_histories[i].is_full():
                                # pad over last block trajectory
                                if last_game_histories[i] is not None:
                                    self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                                # calculate priority
                                priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                                # save block trajectory
                                last_game_histories[i] = game_histories[i]
                                last_game_priorities[i] = priorities

                                # new block trajectory
                                game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                                config=self.config,
                                                                exploration_episode=(i >= exploit_env_nums and self.config.use_deep_exploration))
                                game_histories[i].init(stack_obs_windows[i])

                    for i in range(env_nums):
                        env = envs[i]
                        env.close()

                        if dones[i]:
                            # pad over last block trajectory
                            if last_game_histories[i] is not None:
                                self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories)

                            # store current block trajectory
                            priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                            game_histories[i].game_over()

                            self.put((game_histories[i], priorities))
                            self.free()

                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1
                        else:
                            # if the final game history is not finished, we will not save this data.
                            total_transitions -= len(game_histories[i])

                    # logs
                    visit_entropies = np.array(self_play_visit_entropy).mean()
                    visit_entropies /= max_visit_entropy

                    if self_play_episodes > 0:
                        log_self_play_moves = self_play_moves / self_play_episodes
                        log_self_play_rewards = self_play_rewards / self_play_episodes
                        log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                    else:
                        log_self_play_moves = 0
                        log_self_play_rewards = 0
                        log_self_play_ori_rewards = 0

                    other_dist = {}
                    # send logs
                    self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                    log_self_play_ori_rewards, log_self_play_rewards,
                                                                    self_play_rewards_max, _temperature.mean(),
                                                                    visit_entropies, 0,
                                                                    other_dist)
        except:
            traceback.print_exc()

    def get_ube_predictions(self, hidden_state_roots, reward_hidden_roots, env_nums, model):
        """
            Computes the ube predictions for each action at batched (abstracted) state hidden_state_roots.
            Used for deep_exploration WITHOUT MuExplore, with action selection that is based on Q values rather than
            only counts.

            Parameters
            ----------
            hidden_state_roots: the output abstracted state of initial inference on batched current observations
            reward_hidden_roots: the lstm hidden state on batched current observations
            env_nums: number of parallel batched envs
            model: the model used by selfplay
        """
        ube_predictions = []
        hidden_state_roots = torch.from_numpy(np.asarray(hidden_state_roots)).to(self.device).float()
        hidden_states_c_reward = torch.from_numpy(np.asarray(reward_hidden_roots[0])).to(
            self.device)
        hidden_states_h_reward = torch.from_numpy(np.asarray(reward_hidden_roots[1])).to(
            self.device)
        for action in range(self.config.action_space_size):
            actions = [action for _ in range(env_nums)]
            actions = torch.from_numpy(np.asarray(actions)).to(self.device).unsqueeze(1).long()
            # Predict value-variance prediction for all environments and specific action
            if self.config.amp_type == 'torch_amp':
                with autocast():
                    network_output = model.recurrent_inference(hidden_state_roots,
                                                               (hidden_states_c_reward,
                                                                hidden_states_h_reward),
                                                               actions)
            else:
                network_output = model.recurrent_inference(hidden_state_roots,
                                                           (hidden_states_c_reward,
                                                            hidden_states_h_reward),
                                                           actions)
            ube_predictions.append(network_output.value_variance)

        return np.asarray(ube_predictions)

    def debug_deep_sea(self, model):
        """
           evaluates the results of MCTS over the diagonal of deep_sea/0
        """
        # First, setup observation batches of shape (10, 1, 10, 10)
        env_nums = 10
        zero_obs = np.zeros(shape=(10, 10))
        batched_obs = []
        for i in range(10):
            current_obs = np.zeros(shape=(10, 10))
            current_obs[i, i] = 1
            # if i == 0:
            #     stack_obs = np.stack([zero_obs, zero_obs, zero_obs, current_obs], axis=0)
            # elif i == 1:
            #     one_obs = np.zeros(shape=(10, 10))
            #     one_obs[0, 0] = 1
            #     stack_obs = np.stack([zero_obs, zero_obs, one_obs, current_obs], axis=0)
            # elif i == 2:
            #     one_obs = np.zeros(shape=(10, 10))
            #     one_obs[0, 0] = 1
            #     two_obs = np.zeros(shape=(10, 10))
            #     two_obs[1, 1] = 1
            #     stack_obs = np.stack([zero_obs, one_obs, two_obs, current_obs], axis=0)
            # else:
            #     stack_obs = []
            #     for j in range(3, 0, -1):
            #         obs = np.zeros(shape=(10, 10))
            #         obs[i - j][i - j] = 1
            #         stack_obs.append(obs)
            #     stack_obs.append(current_obs)
            #     stack_obs = np.asarray(stack_obs)
            stack_obs = current_obs[np.newaxis, :]
            batched_obs.append(stack_obs)
        batched_obs = np.asarray(batched_obs)
        stack_obs = torch.from_numpy(batched_obs).to(self.device).float()

        # Second, setup everything to call MCTS
        if self.config.amp_type == 'torch_amp':
            with autocast():
                network_output = model.initial_inference(stack_obs.float())
        else:
            network_output = model.initial_inference(stack_obs.float())
        hidden_state_roots = network_output.hidden_state
        reward_hidden_roots = network_output.reward_hidden
        value_prefix_pool = network_output.value_prefix
        policy_logits_pool = network_output.policy_logits.tolist()
        value_prefix_variance_pool = network_output.value_prefix_variance
        value_pool = network_output.value
        noises = [
            np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(
                np.float32).tolist() for _ in range(env_nums)]
        roots = cytree.Roots(env_nums, self.config.action_space_size, self.config.num_simulations)
        roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool,
                      policy_logits_pool)

        # Call MCTS:
        MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots)
        roots_distributions = roots.get_distributions()
        roots_values = roots.get_values()
        children_values = roots.get_roots_children_values(self.config.discount)

        # Print the results
        for i in range(10):
            action_right = self.visitation_counter.identify_action_right(i, i)
            print(f"At state {(i, i)}, root_value = {roots_values[i]}, prediction_value = {value_pool[i]} children_values = {children_values[i]} roots_distributions = {roots_distributions[i]}, action_right = {action_right}"
                  , flush=True)

    def debug_uncertainty(self, total_transitions, row, column, i, roots, action, distributions, value, deterministic):
        if i > (self.config.p_mcts_num - self.config.number_of_exploratory_envs) and self.config.mu_explore and 'deep_sea' in self.config.env_name \
                and self.config.plan_with_fake_visit_counter and total_transitions > self.config.start_transitions:
            action_right = self.visitation_counter.identify_action_right(row, column)
            children_uncertainties = roots.get_roots_children_uncertainties(self.config.discount)[i]
            # If everything was correct except action selection, print
            if row == column and action_right != action:
                print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n"
                      f"In env {i}, fake uncertainty, action selection is incorrect. \n "
                      f"Action right is {action_right} and action chosen in {action} \n"
                      f"State is {(row, column)} \n"
                      f"Uncertainties are: {children_uncertainties} \n"
                      f"Visitations counting are: {distributions} \n"
                      f"Value is: {value} \n"
                      f"deterministic is: {deterministic} \n"
                      f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n"
                      , flush=True)
            # If action right has LESS uncertainty than action left, and state is along the diagonal, print visitations, uncertainties and values
            if row == column and children_uncertainties[action_right] <= children_uncertainties[1 - action_right]:
                print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n"
                      f"In env {i}, fake uncertainty, MCTS uncertainty is incorrect. \n "
                      f"Action right is {action_right} \n"
                      f"State is {(row, column)} \n"
                      f"Uncertainties are: {children_uncertainties} \n"
                      f"Visitations counting are: {distributions} \n"
                      f"Value is: {value} \n"
                      f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n"
                      , flush=True)
            # If visitations right are LESS than visitations left and state is along the diagonal, print every
            if row == column and np.argmax(distributions) != np.argmax(children_uncertainties):
                print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n"
                      f"In env {i}, visitation counting do not agree with uncertainty max. \n "
                      f"Action right is {action_right} \n"
                      f"Uncertainties are: {children_uncertainties} \n"
                      f"Visitations counting are: {distributions} \n"
                      f"Value is: {value} \n"
                      f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n"
                      , flush=True)