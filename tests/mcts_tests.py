import torch
from core.config import DiscreteSupport
import numpy
import numpy as np
# import core.ctree.cytree as cytree
import tests.ctree_copy_for_testing.cytree as cytree
from config import deepsea, atari
from core import utils
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst
from core.visitation_counter import CountUncertainty
from core.mcts import MCTS
from bsuite.utils import gym_wrapper as bsuite_gym_wrapper
from config.deepsea.env_wrapper import DeepSeaWrapper
from bsuite.environments.deep_sea import DeepSea

class MCTS_Tests:
    def __init__(self):
        self.game_config = atari.AtariConfig()
        self.game_config.stacked_observations = 4
        self.game_config.gray_scale = False
        self.game_config.p_mcts_num = 3
        self.game_config.num_unroll_steps = 1
        self.game_config.batch_size = 16
        self.game_config.value_support = DiscreteSupport(-15, 15, delta=1)
        self.game_config.reward_support = DiscreteSupport(-15, 15, delta=1)
        self.game_config.lstm_hidden_size = 64
        self.proj_hid = 32
        self.proj_out = 32
        self.pred_hid = 16
        self.pred_out = 32
        self.game_config.beta = 10
        self.game_config.seed = 0

        self.game_config.set_game("BreakoutNoFrameskip-v4")
        self.env_num = self.game_config.p_mcts_num
        self.rank = 0
        self.envs = [self.game_config.new_game(self.game_config.seed + (self.rank + 1) * i) for i in range(self.env_num)]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.game_config.get_uniform_network()
        self.model.to(self.device)

        self.action_space_size = self.game_config.action_space_size
        self.num_simulations = self.game_config.num_simulations
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.95
        self.value_delta_max = 0.01
        self.beta = self.game_config.beta
        self.results = None

        self.root_exploration_fraction = 0.25
        self.root_dirichlet_alpha = 0.3

    def test_prepare_explore(self):
        #TODO: Test that the prepare_explore correctly initializes one root that is exploit, and the rest are explore
        raise NotImplementedError

    def test_batch_back_propagate(self):
        #TODO: Test that exploratory batch_back_propagate works as expected

        raise NotImplementedError

    def transform_observation_from_atari_to_model(self, observation):
        # Turn the observation into a numpy array
        observation = utils.str_to_arr(observation)
        # Turn it into the output of step_obs, length of list is index:index + self.stacked_observations
        observation = [observation]
        # turn into the input to prepare_observation_lst, which is a list of length p_mcts_num
        observation = [observation]
        # observation = torch.from_numpy(np.array(observation)).to(self.game_config.device)
        observation = utils.prepare_observation_lst(observation)
        observation = torch.from_numpy(observation).to(self.game_config.device).float() / 255.0
        return observation

    def initial_inference(self):
        init_obses = [env.reset() for env in self.envs]
        dones = np.array([False for _ in range(self.env_num)])
        game_histories = [GameHistory(self.envs[_].env.action_space, max_length=self.game_config.history_length,
                                      config=self.game_config) for _ in range(self.env_num)]

        # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
        stack_obs_windows = [[] for _ in range(self.env_num)]

        for i in range(self.env_num):
            stack_obs_windows[i] = [init_obses[i] for _ in range(self.game_config.stacked_observations)]
            game_histories[i].init(stack_obs_windows[i])

        # stack obs for model inference
        stack_obs = [game_history.step_obs() for game_history in game_histories]
        if self.game_config.image_based:
            stack_obs = prepare_observation_lst(stack_obs)
            stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
        else:
            stack_obs = [game_history.step_obs() for game_history in game_histories]
            stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)
        initial_inference = self.model.initial_inference(stack_obs.float())
        hidden_state_roots = initial_inference.hidden_state
        reward_hidden_roots = initial_inference.reward_hidden
        value_prefix_pool = initial_inference.value_prefix
        policy_logits_pool = initial_inference.policy_logits.tolist()
        value_prefix_variance_pool = initial_inference.value_prefix_variance

        return hidden_state_roots, reward_hidden_roots, value_prefix_pool, policy_logits_pool, value_prefix_variance_pool

    def recurrent_inference(self, hidden_state, hidden_states_c_reward, hidden_states_h_reward, action_batch):
        # action_batch = torch.from_numpy(numpy.asarray(action_batch)).to(self.game_config.device).unsqueeze(-1).long()
        # hidden_state = torch.from_numpy(np.asarray(hidden_state)).to(self.device).float()
        # reward_hidden_c = torch.from_numpy(np.asarray(reward_hidden[0])).to(self.device).float()
        # reward_hidden_h = torch.from_numpy(np.asarray(reward_hidden[1])).to(self.device).float()
        network_output = self.model.recurrent_inference(hidden_state, (hidden_states_c_reward, hidden_states_h_reward), action_batch)

        hidden_state_nodes = network_output.hidden_state
        value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
        value_pool = network_output.value.reshape(-1).tolist()
        policy_logits_pool = network_output.policy_logits.tolist()
        reward_hidden_nodes = network_output.reward_hidden
        value_prefix_variance_pool = network_output.value_prefix_variance.reshape(-1).tolist()
        value_variance_pool = network_output.value_variance.reshape(-1).tolist()

        return hidden_state_nodes, value_prefix_pool, value_pool, policy_logits_pool, reward_hidden_nodes, value_prefix_variance_pool, value_variance_pool

    def back_to_back_test(self):
        print(f"Running back to back, Python->C->Python smoke test")
        self.model.eval()
        # Call initial inference
        print(f"Calling initial_inference")
        hidden_state_roots, reward_hidden_roots, value_prefix_pool, policy_logits_pool, value_prefix_variance_pool = self.initial_inference()
        print(f"Testing the shapes of value_prefix_pool and value_prefix_variance_pool \n"
              f"numpy.shape(value_prefix_pool) = {numpy.shape(value_prefix_pool)} \n"
              f"numpy.shape(value_prefix_variance_pool) = {numpy.shape(value_prefix_variance_pool)} \n"
              f"value_prefix_pool = {value_prefix_pool} \n"
              f"value_prefix_variance_pool = {value_prefix_variance_pool} \n"
              f"policy logits: {policy_logits_pool}")

        policy_logits_pool = [policy_logits_pool[0]] + [np.ones_like(policy_logits_pool[0]).tolist()
                                                        for _ in range(len(policy_logits_pool) - 1)]
        print(f"Normalizing policy logits. Policy logits after normalization: {policy_logits_pool}")

        # Init CRoots
        print(f"Initializing CRoots")
        test_roots = cytree.Roots(self.env_num, self.action_space_size, self.num_simulations, self.beta)

        # Call prepare explore
        noises = [np.random.dirichlet([self.root_dirichlet_alpha] * self.action_space_size).astype(
            np.float32).tolist() for _ in range(self.env_num)]

        print(f"Calling prepare_explore")
        test_roots.prepare_explore(self.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool,
                                        value_prefix_variance_pool, self.beta)

        # Preparation for MCTS
        print(f"Preparing for MCTS")
        self.min_max_stats_lst = cytree.MinMaxStatsList(test_roots.num)
        hidden_state_pool = [hidden_state_roots]
        # 1 x batch x 64
        # the data storage of value prefix hidden states in LSTM
        reward_hidden_c_pool = [reward_hidden_roots[0]]
        reward_hidden_h_pool = [reward_hidden_roots[1]]
        # the index of each layer in the tree
        hidden_state_index_x = 0
        horizons = self.game_config.lstm_horizon_len
        self.min_max_stats_lst.set_delta(self.value_delta_max)

        hidden_states = []
        hidden_states_c_reward = []
        hidden_states_h_reward = []
        self.results = cytree.ResultsWrapper(test_roots.num)

        # Call traverse
        print(f"Calling batch_traverse")
        # traverse to select actions for each root
        # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
        # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
        # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
        hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = cytree.batch_traverse(test_roots,
                                                                                                 self.pb_c_base,
                                                                                                 self.pb_c_init,
                                                                                                 self.discount,
                                                                                                 self.min_max_stats_lst,
                                                                                                 self.results)
        search_lens = self.results.get_search_len()

        # Prepare for recurrent inf
        print(f"Preparing for recurrent_inference")
        # obtain the states for leaf nodes
        for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
            hidden_states.append(hidden_state_pool[ix][iy])
            hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
            hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

        hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(self.device).float()
        hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(self.device).unsqueeze(0)
        hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(self.device).unsqueeze(0)

        last_actions = torch.from_numpy(np.asarray(last_actions)).to(self.device).unsqueeze(1).long()

        # Call recurrent_inference
        print(f"Calling recurrent_inference")
        hidden_state_nodes, value_prefix_pool, value_pool, policy_logits_pool, reward_hidden_nodes, \
            value_prefix_variance_pool, value_variance_pool = \
            self.recurrent_inference(hidden_states, hidden_states_c_reward, hidden_states_h_reward, last_actions)

        # Prepare for batch backprop
        print(f"Preparing for uncertainty_batch_back_propagate")
        hidden_state_pool.append(hidden_state_nodes)
        # reset 0
        # reset the hidden states in LSTM every horizon steps in search
        # only need to predict the value prefix in a range (eg: s0 -> s5)
        assert horizons > 0
        reset_idx = (np.array(search_lens) % horizons == 0)
        assert len(reset_idx) == test_roots.num
        reward_hidden_nodes[0][:, reset_idx, :] = 0
        reward_hidden_nodes[1][:, reset_idx, :] = 0
        is_reset_lst = reset_idx.astype(np.int32).tolist()

        reward_hidden_c_pool.append(reward_hidden_nodes[0])
        reward_hidden_h_pool.append(reward_hidden_nodes[1])
        hidden_state_index_x += 1

        # Call batch backprop
        print(f"Calling uncertainty_batch_back_propagate")
        cytree.uncertainty_batch_back_propagate(hidden_state_index_x, self.discount,
                                                value_prefix_pool, value_pool, policy_logits_pool,
                                                self.min_max_stats_lst, self.results, is_reset_lst,
                                                value_prefix_variance_pool, value_variance_pool)
        print(f"back to back, Python->C->Python smoke test passed")

        print(f"test_roots.get_roots_children_uncertainties() = {test_roots.get_roots_children_uncertainties(0.997)}")
        print(f"test_roots.get_roots_children_values() = {test_roots.get_roots_children_values(0.997)}")
        # Test that has not crashed

    def test_equivalence_standard_exploratory(self):
        print(f"Running equivalence between  test")
        self.model.eval()
        # Call initial inference
        print(f"Calling initial_inference")
        hidden_state_roots, reward_hidden_roots, value_prefix_pool, policy_logits_pool, value_prefix_variance_pool = self.initial_inference()

        policy_logits_pool_uniformed = [policy_logits_pool[0]] + [np.ones_like(policy_logits_pool[0]).tolist() for _ in
                                                        range(len(policy_logits_pool) - 1)]

        # Init CRoots
        print(f"Initializing CRoots")
        test_roots_exploratory = cytree.Roots(self.env_num, self.action_space_size, self.num_simulations, self.beta)
        test_roots_standard = cytree.Roots(self.env_num, self.action_space_size, self.num_simulations)

        # Call prepare explore
        noises = [np.random.dirichlet([self.root_dirichlet_alpha] * self.action_space_size).astype(
            np.float32).tolist() for _ in range(self.env_num)]

        print(f"Calling prepare_explore")
        test_roots_exploratory.prepare_explore(self.root_exploration_fraction, noises, value_prefix_pool,
                                      policy_logits_pool, value_prefix_variance_pool, self.beta)
        test_roots_standard.prepare(self.root_exploration_fraction, noises, value_prefix_pool,
                                      policy_logits_pool)

        # Preparation for MCTS
        print(f"Preparing for MCTS")
        self.min_max_stats_lst_exploratory = cytree.MinMaxStatsList(test_roots_exploratory.num)
        self.min_max_stats_lst_standard = cytree.MinMaxStatsList(test_roots_standard.num)
        hidden_state_pool = [hidden_state_roots]
        # 1 x batch x 64
        # the data storage of value prefix hidden states in LSTM
        reward_hidden_c_pool = [reward_hidden_roots[0]]
        reward_hidden_h_pool = [reward_hidden_roots[1]]
        # the index of each layer in the tree
        hidden_state_index_x = 0
        horizons = self.game_config.lstm_horizon_len
        self.min_max_stats_lst_exploratory.set_delta(self.value_delta_max)
        self.min_max_stats_lst_standard.set_delta(self.value_delta_max)

        hidden_states = []
        hidden_states_c_reward = []
        hidden_states_h_reward = []
        self.results_exploratory = cytree.ResultsWrapper(test_roots_exploratory.num)
        self.results_standard = cytree.ResultsWrapper(test_roots_standard.num)

        # Call traverse
        print(f"Calling batch_traverse")
        # traverse to select actions for each root
        # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
        # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
        # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
        hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = cytree.batch_traverse(test_roots_exploratory,
                                                                                                 self.pb_c_base,
                                                                                                 self.pb_c_init,
                                                                                                 self.discount,
                                                                                                 self.min_max_stats_lst_exploratory,
                                                                                                 self.results_exploratory)
        hidden_state_index_x_lst_standard, hidden_state_index_y_lst_standard, last_actions_standard = cytree.batch_traverse(test_roots_standard,
                                                                                                 self.pb_c_base,
                                                                                                 self.pb_c_init,
                                                                                                 self.discount,
                                                                                                 self.min_max_stats_lst_standard,
                                                                                                 self.results_standard)
        print(f"Testing that output of batch_traverse is the same for both roots")
        assert (hidden_state_index_x_lst == hidden_state_index_x_lst_standard and hidden_state_index_y_lst == hidden_state_index_y_lst_standard and last_actions == last_actions)
        print(f"Equivalence test passed")
        search_lens = self.results_exploratory.get_search_len()

        # Prepare for recurrent inf
        print(f"Preparing for recurrent_inference")
        # obtain the states for leaf nodes
        for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
            hidden_states.append(hidden_state_pool[ix][iy])
            hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
            hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])

        hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(self.device).float()
        hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(self.device).unsqueeze(0)
        hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(self.device).unsqueeze(0)

        last_actions = torch.from_numpy(np.asarray(last_actions)).to(self.device).unsqueeze(1).long()

        # Call recurrent_inference
        print(f"Calling recurrent_inference")
        hidden_state_nodes, value_prefix_pool, value_pool, policy_logits_pool, reward_hidden_nodes, \
            value_prefix_variance_pool, value_variance_pool = \
            self.recurrent_inference(hidden_states, hidden_states_c_reward, hidden_states_h_reward, last_actions)

        print(f"value_prefix_variance_pool = {value_prefix_variance_pool} \n"
              f"value_variance_pool = {value_variance_pool}")

        # Prepare for batch backprop
        print(f"Preparing for uncertainty_batch_back_propagate")
        hidden_state_pool.append(hidden_state_nodes)
        # reset 0
        # reset the hidden states in LSTM every horizon steps in search
        # only need to predict the value prefix in a range (eg: s0 -> s5)
        assert horizons > 0
        reset_idx = (np.array(search_lens) % horizons == 0)
        assert len(reset_idx) == test_roots_exploratory.num
        reward_hidden_nodes[0][:, reset_idx, :] = 0
        reward_hidden_nodes[1][:, reset_idx, :] = 0
        is_reset_lst = reset_idx.astype(np.int32).tolist()

        reward_hidden_c_pool.append(reward_hidden_nodes[0])
        reward_hidden_h_pool.append(reward_hidden_nodes[1])
        hidden_state_index_x += 1

        # Call batch backprop
        print(f"Calling uncertainty_batch_back_propagate")
        cytree.uncertainty_batch_back_propagate(hidden_state_index_x, self.discount,
                                                value_prefix_pool, value_pool, policy_logits_pool,
                                                self.min_max_stats_lst_exploratory, self.results_exploratory, is_reset_lst,
                                                value_prefix_variance_pool, value_variance_pool)
        cytree.batch_back_propagate(hidden_state_index_x, self.discount,
                                                value_prefix_pool, value_pool, policy_logits_pool,
                                                self.min_max_stats_lst_standard, self.results_standard, is_reset_lst)
        print(f"Testing equivalence of uncertainty_batch_back_propagate")
        # check that all the values of all the nodes are the same
        exploratory_roots_values = test_roots_exploratory.get_values()
        standard_roots_values = test_roots_standard.get_values()
        assert standard_roots_values == exploratory_roots_values
        print(f"Equivalence of uncertainty_batch_back_propagate test passed")
        print(f"Testing values_uncertainty. \n"
              f"test_roots_exploratory.get_values_uncertainty() = {test_roots_exploratory.get_values_uncertainty()}")
        print(f"uncertainty_batch_back_propagate smoke test passed")

    def test_MCTS_w_counter_uncertainty(self):
        # Init params
        self.config = deepsea.DeepSeaConfig()
        self.config.use_uncertainty_architecture = True
        self.config.ensemble_size = 3
        self.config.stacked_observations = 4
        self.config.p_mcts_num = 2
        self.config.num_unroll_steps = 5
        self.config.batch_size = 2
        self.config.value_support = DiscreteSupport(-5, 5, delta=1)
        self.config.reward_support = DiscreteSupport(-5, 5, delta=1)
        self.config.lstm_hidden_size = 16
        self.config.mu_explore = True
        self.config.plan_with_visitation_counter = True
        self.config.use_visitation_counter = True
        self.config.proj_hid = 32
        self.config.proj_out = 32
        self.config.pred_hid = 16
        self.config.pred_out = 32
        self.config.seed = 5
        self.config.beta = 100
        self.config.root_exploration_fraction = 0
        self.config.amp_type = 'torch_amp'

        self.config.set_game("deep_sea/0")

        self.env = self.config.new_game()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Init a NN
        self.model = self.config.get_uniform_network()
        self.model.to(self.device)
        self.model.eval()
        # Init state counter with fake unc.
        self.visitation_counter = CountUncertainty(name=self.config.env_name, num_envs=2, mapping_seed=self.config.seed, fake=True)
        use_state_visits = True
        num_episodes = 50
        num_actions_per_episode = 10
        env = bsuite_gym_wrapper.GymFromDMEnv(
            DeepSea(size=10, mapping_seed=self.config.seed, seed=self.config.seed))
        env = DeepSeaWrapper(env, discount=0.997, cvt_string=False)
        for i in range(num_episodes):
            observation = env.reset()
            for j in range(num_actions_per_episode):
                # Init stacked observations - start with first observation
                stack_obs = np.repeat(observation[np.newaxis, :, :], self.config.stacked_observations, axis=0)
                stack_obs = np.repeat(stack_obs[np.newaxis, :, :], self.config.p_mcts_num, axis=0)
                initial_observations_for_counter = self.visitation_counter.from_one_hot_state_to_indexes(
                    np.array(stack_obs, dtype=np.uint8)[:, -1, :, :])
                stack_obs = torch.from_numpy(stack_obs).to(self.device).float()
                print(stack_obs.shape)
                # Call initial inference
                initial_inference = self.model.initial_inference(stack_obs)
                hidden_state_roots = initial_inference.hidden_state
                print(np.shape(initial_inference.hidden_state))
                exit()
                reward_hidden_roots = initial_inference.reward_hidden
                value_prefix_pool = initial_inference.value_prefix
                policy_logits_pool = initial_inference.policy_logits.tolist()
                value_prefix_variance_pool = initial_inference.value_prefix_variance
                noises = [
                    np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(
                        np.float32).tolist() for _ in range(self.config.p_mcts_num)]
                policy_logits_pool = [
                                         policy_logits_pool[0]] + [np.ones_like(policy_logits_pool[0]).tolist()
                                         for _ in range(len(policy_logits_pool) - 1)
                                        ]
                roots = cytree.Roots(self.config.p_mcts_num, self.config.action_space_size, self.config.num_simulations, self.config.beta)
                roots.prepare_explore(self.config.root_exploration_fraction, noises, value_prefix_pool,
                                      policy_logits_pool, value_prefix_variance_pool, self.config.beta)
                # Call MCTS with large beta
                MCTS(self.config).search_w_visitation_counter(roots, self.model, hidden_state_roots, reward_hidden_roots,
                                                              self.visitation_counter, initial_observations_for_counter, use_state_visits)
                roots_distributions = roots.get_distributions()
                roots_values = roots.get_values()
                roots_uncertainties = roots.get_values_uncertainty()
                # print(f"roots_distribution = {roots_distributions[1]}, roots_value = {roots_values[1]}, root_uncertainties = {roots_uncertainties[1]}")
                # select action
                action, visit_entropy = select_action(roots_distributions[1], temperature=1,
                                                      deterministic=True)
                # Observe
                self.visitation_counter.observe(observation, action)

                # step
                observation, reward, done, info = env.step(action)
            print(f"In episode: {i}")
            print(f"Last root info: distribution = {roots_distributions[1]}, value = {roots_values[1]}, uncertainty = {roots_uncertainties[1]}")
            print(f"The state counter is: {self.visitation_counter.s_counts}")
            print(f"Last row of state_action counter is: {self.visitation_counter.sa_counts[-1, :, :]}")
            # if (self.visitation_counter.sa_counts[-1, :, :] == 2).any():
            #     exit()

# MCTS_Tests().test_MCTS_w_counter_uncertainty()
MCTS_Tests().back_to_back_test()