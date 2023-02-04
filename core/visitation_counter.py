import numpy as np
from bsuite import sweep
# from bsuite.environments.deep_sea import DeepSea
from config.deepsea.extended_deep_sea import DeepSea

# Taken from Wendelin's implementation in the DRL HW explore
class CountUncertainty:
    """
        Defines an uncertainty estimate based on counts over the state/observation space.
        Uncertainty will be scaled by 'scale'.
        Only implemented for the deep_sea environment
    """
    def __init__(self, name, num_envs, mapping_seed, scale=1, epsilon=1E-7, fake=False):
        """
            name: the name of the env (deep_sea/N)
            num_envs: num of parallel envs for planning with MCTS
            scale: the scale of the uncertainty, uncertainty = unc * scale
            epsilon: the scale of max. uncertainty (zero visitaitons). unc = 1 / (epsilon + visit_count)
            size: size of environment. Observations are square, so they are of shape size x size
            gamma: the discount used is 1
            fake: for debugging - if true, the rewarding transition is always associated with max uncertainty.
        """
        self.name = name
        self.size = sweep.SETTINGS[name]['size']
        self.observation_space_shape = (self.size, self.size)
        self.action_space = 2   # the deepsea action space is 2
        self.s_counts = np.zeros(shape=self.observation_space_shape)
        self.sa_counts = np.zeros(shape=(self.observation_space_shape + (self.action_space,)))
        self.scale = scale
        self.eps = epsilon
        self.num_envs = num_envs
        # we init all envs with the same seed, because we also init all the real envs with the same seed
        self.planning_env = DeepSea(size=self.size, mapping_seed=mapping_seed, seed=mapping_seed)
        self.observation_counter = 0
        self.rewarding_transition = self.identify_rewarding_transition()
        self.fake_reward_uncertainty = fake

    def identify_rewarding_transition(self):
        """
            returns the index of row, column and action of the positive-rewarding transition of deep_sea
        """
        # The best transition is at action right at state -1,-1
        action_right = self.planning_env._action_mapping[-1, -1]
        row_index, column_index = self.observation_space_shape[0] - 1, self.observation_space_shape[1] - 1
        return row_index, column_index, action_right

    def identify_action_right(self, row, column):
        """
            For a state in form row / column, returns which action is right (0 or 1)
        """
        return self.planning_env._action_mapping[row, column]

    @staticmethod
    def from_one_hot_state_to_indexes(state):
        """
            Takes a numpy array, one-hot encoded state of shape (h, w) and returns the indexes of the 1 in the encoding
            If state is the "null state" (all zeros), returns negative indexes
        """
        if len(np.shape(state)) == 2:
            # If the given state is the "null state" (outside of env bounds)
            if not state.any():
                return -1, -1
            else:
                indexes = (state == 1).nonzero()
                index_row, index_column = indexes[0][0], indexes[1][0]
                return index_row, index_column
        elif len(np.shape(state)) == 3:
            # If the given state is the "null state" (outside of env bounds)
            indexes_of_states = []
            for i in range(np.shape(state)[0]):
                if not state[i].any():
                    indexes_of_states.append([-1, -1])
                else:
                    indexes = (state[i] == 1).nonzero()
                    index_row, index_column = indexes[0][0], indexes[1][0]
                    indexes_of_states.append([index_row, index_column])
            return indexes_of_states
        else:
            raise ValueError(f"from_one_hot_state_to_indexes is not ")

    def observe(self, state, action):
        """ Add counts for observed 'state' (observations from deep_sea, which are also the true state).
            'state' is a numpy array of shape (height, width)
        """
        if len(np.shape(state)) == 2:
            # The shape of the expected input is:
            # (height, width)
            row, column = self.from_one_hot_state_to_indexes(state)
            if row < 0 or column < 0:
                raise ValueError(f"Tried to observe the 'null' state. State was: {state}")
            self.s_counts[row, column] += 1
            self.sa_counts[row, column, action] += 1
            self.observation_counter += 1
        # Alternatively, if we got a state in indexes form
        elif len(state) == 2 and not len(np.shape(state)) == 2:
            assert state[0] < self.size and state[0] < self.size
            self.s_counts[state[0], state[1]] += 1
            self.sa_counts[state[0], state[1], action] += 1
            self.observation_counter += 1

    def get_next_true_observation(self, states, actions):
        """"
            Expects a states numpy array of shape (num_envs, w, h) and actions a list of length num_envs
        """
        assert len(np.shape(states)) == 3
        assert len(actions) == np.shape(states)[0] == self.num_envs

        # Identify the rows and columns for each state
        indexes = [(self.from_one_hot_state_to_indexes(states[i])) for i in range(self.num_envs)]
        next_observations = []
        for i in range(self.num_envs):
            # If this is the null state, can just retain the null state
            if indexes[i][0] < 0 or indexes[i][1] < 0:
                next_observations.append(states[i])
            else:
                # set the envs to the right state
                self.planning_env.set_state(indexes[i][0], indexes[i][1])
                row, _ = self.planning_env.get_state()
                if row <  self.size:
                    # get them to step to the next state
                    _ = self.planning_env._step(actions[i])
                # get the observation
                next_observations.append(self.planning_env._get_observation())

        # stack them in dim 0
        next_observations = np.stack(next_observations, axis=0)

        return next_observations

    def get_next_true_observation_indexes(self, indexes, actions):
        """"
            Expects a indexes numpy array of shape (num_envs, 2) and actions a list of length num_envs
        """
        # If shape of indexes = (num_envs, 2)
        if len(np.shape(indexes)) == 2:
            assert len(actions) == np.shape(indexes)[0] == self.num_envs
            # Identify the rows and columns for each state
            next_observations_indexes = []
            for i in range(self.num_envs):
                # If this is the last state or the null state append null state
                if indexes[i][0] >= self.size:
                    next_observations_indexes.append([indexes[i][0], indexes[i][1]])
                else:
                    # set the envs to the right state
                    self.planning_env.set_state(indexes[i][0], indexes[i][1])
                    # get them to step to the next state
                    _ = self.planning_env._step(actions[i])
                    # get the indexes
                    next_observations_indexes.append(self.planning_env.get_state())
            # stack them in dim 0
            next_observations_indexes = np.asarray(next_observations_indexes)
            return next_observations_indexes
        # alternatively, if indexes is just one list of [index_row, index_col]
        elif len(indexes) == 2:
            # If already out of env, return the observation
            if indexes[0] >= self.size:
                return indexes[0], indexes[1]
            else:
                # set the envs to the right state
                self.planning_env.set_state(indexes[0], indexes[1])
                # get them to step to the next state
                _ = self.planning_env._step(actions)
                # get the indexes
                return self.planning_env.get_state()

    def state_action_uncertainty(self, index_row, index_column, action, use_state_visits=True):
        """
            Returns the uncertainty with state-action visitation pair, or with next-state visitations.
            If fake_reward_uncertainty is true, returns max uncertainty with rewarding state always, regardless of count
            If state_visits is True, returns next-state visitations based uncertainty, and otherwise,
                state-action visitation based.
        """
        # If this is the rewarding state
        if self.fake_reward_uncertainty and (index_row, index_column, action) == self.rewarding_transition:
            return self.scale / self.eps
        # If this state is already out of bounds
        elif index_row >= self.size:
            return 0
        # If we want state visits, and this state is not yet out of bounds or last row
        elif use_state_visits and index_row < self.planning_env._size - 1:
            # get next state
            next_state = self.get_next_true_observation_indexes(indexes=[index_row, index_column], actions=action)
            # return the uncertainty of the counter
            return self.scale / (self.s_counts[next_state[0], next_state[1]] + self.eps)
        # Otherwise we return the state-action count. This is either the last row transition, or stat-action uncertainty
        else:
            return self.scale / (self.sa_counts[index_row, index_column, action] + self.eps)

    def get_reward_uncertainty(self, state, action, use_state_visits=False):
        """
            If state is an array of shape (num_envs, height, width) = (4, 10, 10) and action a list of
            len num_envs:
                We return a list of uncertainties, of length num_envs, and all but the last observation in each stack is
                ignored
            If state is a tensor of shape (height, width) and action is a number:
                returns the uncertainty of the state action pair
            Does not change the counters.
        """
        # If the state is just one state in indexes form [row, col]
        if len(np.shape(state)) == 1:
            return self.state_action_uncertainty(state[0], state[1], action, use_state_visits)
        elif np.shape(state) == (self.num_envs, 2):
            reward_uncertainties = []
            assert len(action) == self.num_envs  # verify that the number of actions matches the number of states
            for i in range(self.num_envs):
                reward_uncertainties.append(self.state_action_uncertainty(state[i][0], state[i][1], action[i], use_state_visits=use_state_visits))
            return np.asarray(reward_uncertainties)
        else:
            raise ValueError(f"get_reward_uncertainty is only implemented for states of shape "
                             f"(num_envs, 2) or (2,) = [index_row, index_col], and shape was: {np.shape(state.shape)}")

    def get_surface_value_uncertainty(self, state, use_state_visits=False):
        """
            Implemented for state of shape (num_envs, indexes) = (4, 2), or indexes = [row, col]
                If use_state_visits:
                    Return local state unc. * propagation horizon (which is size - row)
                Else:
                    Return local state_action unc. * propagation horizon (which is size - row)
            If state is a tensor of shape (num_envs, height, width) = (4, 10, 10):
                evaluate the reward-uncertainty for each state and each possible action, take the average and then
                factor the assumption that
            If state is a tensor of shape (height, width):
                evaluate the reward-uncertainty for each action, and return the average
        """
        if np.shape(state) == (self.num_envs, 2):
            # find horizon:
            h = self.size
            # get individual rows from each state:
            current_rows = [state[i][0] for i in range(self.num_envs)]
            # The horizons are size - row if row is in env, otherwise 0
            per_env_horizons = [h - current if current > 0 else 0 for current in current_rows]
            # compute the discount factors for each value-uncertainty
            # discount_factors = np.asarray([(1 - self.gamma ** (2 * horizon)) / (1 - self.gamma ** 2)
            #                     for horizon in per_env_horizons])
            # compute the uncertainties for each action
            first_actions = [0] * self.num_envs
            second_actions = [1] * self.num_envs
            reward_uncertainties_first_action = self.get_reward_uncertainty(state, first_actions, use_state_visits)
            reward_uncertainties_second_action = self.get_reward_uncertainty(state, second_actions, use_state_visits)
            result = per_env_horizons * np.maximum(reward_uncertainties_first_action, reward_uncertainties_second_action)
            return result
        elif len(state) == 2:
            h = self.observation_space_shape[0]
            row = state[0]
            # If this state is outside the bounds of the environment, return 0 uncertainty
            if row < 0:
                return 0
            horizon = h - row
            first_action_reward_unc = self.get_reward_uncertainty(state, 0, use_state_visits)
            second_action_reward_unc = self.get_reward_uncertainty(state, 1, use_state_visits)
            return horizon * max(first_action_reward_unc, second_action_reward_unc)
        else:
            raise ValueError(f"get_surface_value_uncertainty is not implemented for states of shape != (num_envs, height, width) or (height, width), and shape was: {state.shape}")

    def get_propagated_value_uncertainty(self, state, propagation_horizon=3, sampling_times=0, use_state_visits=False):
        """
            Takes an index-type state of shape (num_envs, 2), action, environment and horizon.
            Uses the environment to plan and compute the value uncertainty based on a trajectory of reward uncertainties
            returns the environment as it recieved it
            sampling times:
                If > 0, do stochastic MC up to horizon propagation_horizon of number sampling_times
                If <= 0, do a complete tree of depth propagation_horizon
        """
        assert np.shape(state) == (self.num_envs, 2)
        if sampling_times > 0:
            samples = []
            for i in range(sampling_times):
                propagated_uncertainty = self.sampled_recursive_propagated_uncertainty(np.asarray(state), propagation_horizon, use_state_visits)
                samples.append(propagated_uncertainty)
            propagated_uncertainty = np.stack(samples, axis=0).max(axis=0)
        else:
            propagated_uncertainty = self.recursive_propagated_uncertainty(np.asarray(state), propagation_horizon, use_state_visits)
        return propagated_uncertainty

    def recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits=False):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0 or min(state[:, 0]) >= self.observation_space_shape[0] - 1:
            return np.asarray(self.get_surface_value_uncertainty(state, use_state_visits))
        else:
            left_reward_uncertainty = self.get_reward_uncertainty(state, [0 for _ in range(self.num_envs)], use_state_visits)
            right_reward_uncertainty = self.get_reward_uncertainty(state, [1 for _ in range(self.num_envs)], use_state_visits)
            left_next_state = self.get_next_true_observation_indexes(state, actions=[0 for _ in range(self.num_envs)])
            right_next_state = self.get_next_true_observation_indexes(state, actions=[1 for _ in range(self.num_envs)])
            return np.maximum(
                left_reward_uncertainty + self.recursive_propagated_uncertainty(left_next_state, propagation_horizon - 1, use_state_visits),
                right_reward_uncertainty + self.recursive_propagated_uncertainty(right_next_state, propagation_horizon - 1, use_state_visits),
            )

    def sampled_recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0 or min(state[:, 0]) >= self.observation_space_shape[0]:
            return np.asarray(self.get_surface_value_uncertainty(state, use_state_visits))
        else:
            actions = [np.random.randint(low=0,high=2) for _ in range(self.num_envs)]
            local_reward_uncertainty = self.get_reward_uncertainty(state, actions, use_state_visits)
            sampled_next_state = self.get_next_true_observation_indexes(state, actions=actions)
            return local_reward_uncertainty + self.sampled_recursive_propagated_uncertainty(sampled_next_state, propagation_horizon - 1, use_state_visits)
