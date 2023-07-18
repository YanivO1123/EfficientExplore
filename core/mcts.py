import torch

import numpy as np
import core.ctree.cytree as tree

from core.visitation_counter import CountUncertainty

from torch.cuda.amp import autocast as autocast


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, reward_hidden_roots, acting=False, propagating_uncertainty=False,
                                    recurrent_rnd_hidden_state_roots=None):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        model: Any
            the NN of MuZero
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        acting: bool
            Distinguishes between acting in the environment (True) and training (False)
        propagating_uncertainty: bool
            If true, instead of running MCTS with rewards and values, runs MCTS with
            reward_uncertainty and value_uncertainty (and discount ** 2 instead of discount)
        """
        with torch.no_grad():
            model.eval()
            # preparation
            num = roots.num
            num_exploratory = self.config.number_of_exploratory_envs
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, 1 #self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                recurrent_rnd_hidden_state_pool_1 = [recurrent_rnd_hidden_state_roots[0]]
                recurrent_rnd_hidden_state_pool_2 = [recurrent_rnd_hidden_state_roots[1]]

            for index_simulation in range(self.config.num_simulations if not propagating_uncertainty else self.config.num_simulations_ube):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []
                if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                    rnd_hidden_states_1 = []
                    rnd_hidden_states_2 = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])
                    if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                        rnd_hidden_states_1.append(recurrent_rnd_hidden_state_pool_1[ix][iy])
                        rnd_hidden_states_2.append(recurrent_rnd_hidden_state_pool_2[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)
                if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                    rnd_hidden_states_1 = torch.from_numpy(np.asarray(rnd_hidden_states_1)).to(device).float()
                    rnd_hidden_states_2 = torch.from_numpy(np.asarray(rnd_hidden_states_2)).to(device).float()

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward,
                                                                                   hidden_states_h_reward),
                                                                   last_actions,
                                                                   recurrent_rnd_hidden_state=(rnd_hidden_states_1,
                                                                                               rnd_hidden_states_2) if
                                                                   'r_rnd' in self.config.uncertainty_architecture_type
                                                                   and (propagating_uncertainty or acting) else None
                                                                   )
                else:
                    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward,
                                                                               hidden_states_h_reward),
                                                               last_actions,
                                                               recurrent_rnd_hidden_state=(rnd_hidden_states_1,
                                                                                           rnd_hidden_states_2) if
                                                               'r_rnd' in self.config.uncertainty_architecture_type
                                                               and (propagating_uncertainty or acting) else None
                                                               )

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden
                value_prefix_variance_pool = network_output.value_prefix_variance.reshape(-1).tolist() if network_output.value_prefix_variance is not None else None
                value_variance_pool = network_output.value_variance.reshape(-1).tolist() if network_output.value_variance is not None else None
                if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                    recurrent_rnd_hidden_state_nodes = network_output.recurrent_rnd_hidden_state

                hidden_state_pool.append(hidden_state_nodes)
                if 'r_rnd' in self.config.uncertainty_architecture_type and (propagating_uncertainty or acting):
                    recurrent_rnd_hidden_state_pool_1.append(recurrent_rnd_hidden_state_nodes[0])
                    recurrent_rnd_hidden_state_pool_2.append(recurrent_rnd_hidden_state_nodes[1])
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                reward_hidden_nodes[0][:, reset_idx, :] = 0
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                #MuExplore: Backprop. w. uncertainty
                if self.config.mu_explore and acting and self.config.use_uncertainty_architecture:
                    if self.config.disable_policy_in_exploration:
                        len_logits = len(policy_logits_pool[0])
                        policy_logits_pool = [policy_logits_pool[0]] + [[1.0] * len_logits
                                                                        for _ in range(len(policy_logits_pool) - 1)]
                    tree.uncertainty_batch_back_propagate(hidden_state_index_x, discount,
                                              value_prefix_pool, value_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst,
                                              value_prefix_variance_pool, value_variance_pool, num_exploratory)
                # If we are calling search from reanalyze to generate new UBE targets, we propagate uncertainty instead
                # of values and rewards
                elif self.config.mu_explore and self.config.use_uncertainty_architecture and propagating_uncertainty \
                        and not acting:
                    assert value_prefix_variance_pool is not None and value_variance_pool is not None
                    # When we call MCTS from reanalyze to make UBE targets, we are not interested in the value
                    # uncertainty of the tree following the policy, but rather, the max
                    len_logits = len(policy_logits_pool[0])
                    policy_logits_pool = [[1.0] * len_logits for _ in range(len(policy_logits_pool))]
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount ** 2,
                                              value_prefix_variance_pool, value_variance_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)
                else:
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount,
                                              value_prefix_pool, value_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)

    def search_w_visitation_counter(self, roots, model, hidden_state_roots, reward_hidden_roots, visitation_counter: CountUncertainty,
               initial_observation_roots, use_state_visits=False, sampling_times=0, propagating_uncertainty=False):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        model: Any
            the NN of MuZero
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        visitation_counter: CountUncertainty
            a visitation counter that is updated by the selfplayworker used as a source of uncertainty. Only implemented
            for the deep_sea environment
        initial_observation_roots:
            The numpy array of initial observations for the roots, of shape: (num_roots, height, width)
             for the planner to keep track of true states in planning, to be used with the state counter uncertainty.
        """
        with torch.no_grad():
            model.eval()
            # preparation
            num = roots.num
            num_exploratory = self.config.number_of_exploratory_envs
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]

            # MuExplore: visitation counter: keep track of the true state for planning with the visit counter
            true_observation_pool = [initial_observation_roots]
            value_propagation_horizon = visitation_counter.observation_space_shape[0]

            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            for index_simulation in range(self.config.num_simulations if not propagating_uncertainty else self.config.num_simulations_ube):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # MuExplore
                true_observations = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

                    # Visitation counter
                    true_observations.append(true_observation_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                # MuExplore: Visitation counter
                true_observations = np.asarray(true_observations)
                # Compute the next true observation and keep track of it
                true_observations_nodes = visitation_counter.get_next_true_observation_indexes(true_observations, last_actions)

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                else:
                    network_output = model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                # MuExplore: Compute the uncertainties
                # If we use visitation counter AND ube the value unc. is the sum of surface count-value unc. +
                if self.config.plan_with_visitation_counter and 'ube' in self.config.uncertainty_architecture_type:
                    value_prefix_variance_pool = visitation_counter.get_reward_uncertainty(true_observations,
                                                                                           last_actions,
                                                                                           use_state_visits=use_state_visits).tolist()
                    value_variance_pool = visitation_counter.get_surface_value_uncertainty(true_observations,
                                                                                              use_state_visits=use_state_visits)
                    value_variance_pool = np.maximum(value_variance_pool, network_output.value_variance.reshape(-1)).tolist()
                # Otherwise, if we use visitation counter use propagated value unc. estimate from the visitation count
                elif self.config.plan_with_visitation_counter:
                    value_prefix_variance_pool = visitation_counter.get_reward_uncertainty(true_observations,
                                                                                           last_actions,
                                                                                           use_state_visits=use_state_visits).tolist()
                    # Compute the PROPAGATED value uncertainty, by doing Monte-Carlo sims with the real model for sampling_times sims, up to horizon propagation_horizon
                    value_variance_pool = visitation_counter.get_propagated_value_uncertainty(true_observations,
                                                                                              propagation_horizon=value_propagation_horizon,
                                                                                              sampling_times=sampling_times,
                                                                                              use_state_visits=use_state_visits).tolist()
                # If we don't plan with the visitation count, use the output of the network
                else:
                    value_prefix_variance_pool = network_output.value_prefix_variance.reshape(-1).tolist() if network_output.value_prefix_variance is not None else None
                    value_variance_pool = network_output.value_variance.reshape(-1).tolist() if network_output.value_variance is not None else None

                # MuExplore: state counter, keep track of the true states of the environment
                true_observation_pool.append(true_observations_nodes)

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                reward_hidden_nodes[0][:, reset_idx, :] = 0
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                #MuExplore: Backprop. w. uncertainty
                if self.config.mu_explore and not propagating_uncertainty:
                    if self.config.disable_policy_in_exploration:
                        len_logits = len(policy_logits_pool[0])
                        policy_logits_pool = [policy_logits_pool[0]] + [[1.0] * len_logits for _ in range(len(policy_logits_pool) - 1)]
                    tree.uncertainty_batch_back_propagate(hidden_state_index_x, discount,
                                              value_prefix_pool, value_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst,
                                              value_prefix_variance_pool, value_variance_pool, num_exploratory)
                elif self.config.mu_explore and propagating_uncertainty:
                    assert value_prefix_variance_pool is not None and value_variance_pool is not None
                    # When we call MCTS from reanalyze to make UBE targets, we are not interested in the value
                    # uncertainty of the tree following the policy, but rather, the max
                    len_logits = len(policy_logits_pool[0])
                    policy_logits_pool = [[1.0] * len_logits for _ in range(len(policy_logits_pool))]
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount ** 2,
                                              value_prefix_variance_pool, value_variance_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)
                else:
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount,
                                              value_prefix_pool, value_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)

    def search_w_forward_propagation(self, roots, model, hidden_state_roots, reward_hidden_roots, visitation_counter: CountUncertainty,
               initial_observation_roots, use_state_visits=False, propagating_uncertainty=False,
                                     recurrent_rnd_hidden_state=None):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
                Parameters
                ----------
                roots: Any
                    a batch of expanded root nodes
                model: Any
                    the NN of MuZero
                hidden_state_roots: list
                    the hidden states of the roots
                reward_hidden_roots: list
                    the value prefix hidden states in LSTM of the roots
                visitation_counter: CountUncertainty
                    a visitation counter that is updated by the selfplayworker used as a source of uncertainty. Only implemented
                    for the deep_sea environment
                initial_observation_roots:
                    The numpy array of initial observations for the roots, of shape: (num_roots, height, width)
                     for the planner to keep track of true states in planning, to be used with the state counter uncertainty.
                """
        with torch.no_grad():
            model.eval()
            # preparation
            num = roots.num
            num_exploratory = self.config.number_of_exploratory_envs
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]

            # MuExplore: visitation counter: keep track of the true state for planning with the visit counter
            true_observation_pool = [initial_observation_roots]
            value_propagation_horizon = visitation_counter.observation_space_shape[0]

            # MuExplore forward propagation: store the state-variance matrix
            # Shape needs to be [num_exploratory_roots, state_size, state_size]
            if self.config.learned_model and 'learned' in self.config.representation_type:
                state_variance_roots = [np.zeros(shape=(hidden_state_roots.shape[-1] * hidden_state_roots.shape[-1],
                                                         hidden_state_roots.shape[-1] * hidden_state_roots.shape[-1]))
                                        for _ in range(num)]
            elif self.config.learned_model:
                state_variance_roots = [np.zeros(shape=(hidden_state_roots.shape[-1],
                                                        hidden_state_roots.shape[-1]))
                                        for _ in range(num)]
            else:
                raise ValueError(f"Forward propagation of uncertainty only makes sense if the model is learned.")

            state_variance_pool = [state_variance_roots]

            # 1 x batch x 64
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.config.value_delta_max)
            horizons = self.config.lstm_horizon_len

            for index_simulation in range(
                    self.config.num_simulations if not propagating_uncertainty else self.config.num_simulations_ube):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # MuExplore
                true_observations = []
                state_variances = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.ResultsWrapper(num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base,
                                                                                                       pb_c_init,
                                                                                                       discount,
                                                                                                       min_max_stats_lst,
                                                                                                       results)
                # obtain the search horizon for leaf nodes
                search_lens = results.get_search_len()

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

                    # Visitation counter
                    true_observations.append(true_observation_pool[ix][iy])

                    # Forward propagation:
                    state_variances.append(state_variance_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                # MuExplore: Visitation counter
                true_observations = np.asarray(true_observations)
                # Compute the next true observation and keep track of it
                true_observations_nodes = visitation_counter.get_next_true_observation_indexes(true_observations,
                                                                                               last_actions)

                # MuExplore: forward propagation
                state_variances_nodes = torch.from_numpy(np.asarray(state_variances)).to(device).float()

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.recurrent_inference(hidden_states,
                                                                   (hidden_states_c_reward, hidden_states_h_reward),
                                                                   last_actions)
                else:
                    network_output = model.recurrent_inference(hidden_states,
                                                               (hidden_states_c_reward, hidden_states_h_reward),
                                                               last_actions)

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                # MuExplore: Compute the uncertainties
                # If we use visitation counter AND ube the value unc. is the sum of surface count-value unc. +
                if self.config.plan_with_visitation_counter and 'ube' in self.config.uncertainty_architecture_type:
                    # Compute the local variances
                    local_reward_variance_pool = visitation_counter.get_reward_uncertainty(true_observations,
                                                                                           last_actions,
                                                                                           use_state_visits=use_state_visits).tolist()
                    # High on first observed states
                    local_value_variance_pool_from_counter = visitation_counter.get_surface_value_uncertainty(true_observations,
                                                                                           use_state_visits=use_state_visits)
                    # High on previously observed states
                    local_value_variance_pool_from_ube = network_output.value_variance.reshape(-1)
                    local_value_variance_pool = np.maximum(local_value_variance_pool_from_counter,
                                                           local_value_variance_pool_from_ube).tolist()

                    # Compute reward, value, state variance using the local variances
                    next_state_covariance_matrix_pool, value_prefix_variance_pool, value_variance_pool = self.compute_reward_value_state_variance(
                        model, local_reward_variance_pool, local_value_variance_pool, state_variances_nodes,
                        local_reward_variance_pool, hidden_states, last_actions, hidden_state_nodes, device)
                # Otherwise, if we use visitation counter use propagated value unc. estimate from the visitation count
                elif self.config.plan_with_visitation_counter:
                    raise ValueError(f"Forward propagation of uncertainty only works with UBE")
                # If we don't plan with the visitation count, use the output of the network
                else:
                    # Compute the local variances
                    local_reward_variance_pool = network_output.value_prefix_variance.reshape(
                        -1).tolist() if network_output.value_prefix_variance is not None else None
                    local_value_variance_pool = network_output.value_variance.reshape(
                        -1).tolist() if network_output.value_variance is not None else None

                    # Compute reward, value, state variance using the local variances
                    next_state_covariance_matrix_pool, value_prefix_variance_pool, value_variance_pool = self.compute_reward_value_state_variance(
                        model, local_reward_variance_pool, local_value_variance_pool, state_variances_nodes,
                        local_reward_variance_pool, hidden_states, last_actions, hidden_state_nodes, device)

                # MuExplore: state counter, keep track of the true states of the environment
                true_observation_pool.append(true_observations_nodes)
                state_variance_pool.append(next_state_covariance_matrix_pool)

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                assert len(reset_idx) == num
                reward_hidden_nodes[0][:, reset_idx, :] = 0
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()

                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                # MuExplore: Backprop. w. uncertainty
                if self.config.mu_explore and not propagating_uncertainty:
                    if self.config.disable_policy_in_exploration:
                        len_logits = len(policy_logits_pool[0])
                        policy_logits_pool = [policy_logits_pool[0]] + [[1.0] * len_logits for _ in
                                                                        range(len(policy_logits_pool) - 1)]
                    tree.uncertainty_batch_back_propagate(hidden_state_index_x, discount,
                                                          value_prefix_pool, value_pool, policy_logits_pool,
                                                          min_max_stats_lst, results, is_reset_lst,
                                                          value_prefix_variance_pool, value_variance_pool,
                                                          num_exploratory)
                elif self.config.mu_explore and propagating_uncertainty:
                    assert value_prefix_variance_pool is not None and value_variance_pool is not None
                    # When we call MCTS from reanalyze to make UBE targets, we are not interested in the value
                    # uncertainty of the tree following the policy, but rather, the max
                    len_logits = len(policy_logits_pool[0])
                    policy_logits_pool = [[1.0] * len_logits for _ in range(len(policy_logits_pool))]
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount ** 2,
                                              value_prefix_variance_pool, value_variance_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)
                else:
                    # backpropagation along the search path to update the attributes
                    tree.batch_back_propagate(hidden_state_index_x, discount,
                                              value_prefix_pool, value_pool, policy_logits_pool,
                                              min_max_stats_lst, results, is_reset_lst)

    def compute_total_variance(self, state_covariance_matrix, local_variance, jacobian):
        """
            Computs the sum of: local variance + J Sigma_k J^T, to get variance_r / _v
        """
        first_product = torch.bmm(jacobian, state_covariance_matrix)
        forward_uncertainty = torch.bmm(first_product, jacobian.transpose(-1, -2))
        uncertainty_score = local_variance.squeeze() + forward_uncertainty.squeeze()
        return uncertainty_score

    def compute_reward_value_state_variance(self, model, local_reward_variance, local_value_variance,
                                            state_covariance_matrix, local_transition_variance, state, action,
                                            next_state, device):
        """
            Computes the independent variances: variance_r variance_v and covariance_s
        """
        # Transform everything to torch for computation
        # State and action are already tensors on the right device
        state = state.reshape(next_state.shape[0], -1)
        next_state = torch.from_numpy(np.asarray(next_state)).to(device).float()
        next_state = next_state.reshape(next_state.shape[0], -1)

        local_transition_variance = torch.from_numpy(np.asarray(local_transition_variance)).to(device).float().unsqueeze(-1).unsqueeze(-1)
        local_value_variance = torch.from_numpy(np.asarray(local_value_variance)).to(device).float().unsqueeze(-1)
        local_reward_variance = torch.from_numpy(np.asarray(local_reward_variance)).to(device).float().unsqueeze(-1)

        local_transition_covariance = local_transition_variance * \
                                      torch.eye(state_covariance_matrix.shape[-1]).to(device)

        # Compute jacobians
        J_f, J_r, J_v = model.get_jacobians(state, action, next_state)

        next_state_covariance_matrix = self.compute_total_variance(state_covariance_matrix,
                                                                   local_transition_covariance, J_f)
        reward_variance = self.compute_total_variance(state_covariance_matrix, local_reward_variance, J_r)
        value_variance = self.compute_total_variance(state_covariance_matrix, local_value_variance, J_v)

        return next_state_covariance_matrix.cpu().numpy(), reward_variance.cpu().numpy().tolist(), value_variance.cpu().numpy().tolist()

