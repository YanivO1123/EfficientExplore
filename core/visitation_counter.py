import numpy as np
from bsuite import sweep
from bsuite.environments.deep_sea import DeepSea


# Taken from Wendelin's implementation in the DRL HW explore
class CountUncertainty:
    """ Defines an uncertainty estimate based on counts over the state/observation space.
        Uncertainty will be scaled by 'scale'. Define boundaries either by 'state_bounds'
        or automatically by passing the environment 'env'. The counts will use
        'resolution'^m different bins for m-dimensional state vecotrs"""
    def __init__(self, name, num_envs, mapping_seeds, scale=1, epsilon=1E-7, gamma=0.95):
        """
            name: the name of the env (deep_sea/N)
            num_envs: num of parallel envs for planning with MCTS
            scale: the scale of the uncertainty, uncertainty = unc * scale
            epsilon: the scale of max. uncertainty (zero visitaitons). unc = 1 / (epsilon + visit_count)
            size: size of environment. Observations are square, so they are of shape size x size
            gamma: the discount used to estimate the value uncertainty from reward uncertainties
        """
        self.name = name
        size = sweep.SETTINGS[name]['size']
        self.observation_space_shape = (size, size)
        self.action_space = 2   # the deepsea action space is 2
        self.s_counts = np.zeros(shape=self.observation_space_shape)
        self.sa_counts = np.zeros(shape=(self.observation_space_shape + (self.action_space,)))
        self.scale = scale
        self.eps = epsilon
        self.gamma = gamma
        self.num_envs = num_envs
        self.planning_envs = [DeepSea(size=size, mapping_seed=mapping_seeds[i]) for i in range(num_envs)]
        self.observation_counter = 0

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
            self.s_counts[state[0], state[1]] += 1
            self.sa_counts[state[0], state[1], action] += 1
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
                self.planning_envs[i]._row = indexes[i][0]
                self.planning_envs[i]._column = indexes[i][1]

                if self.planning_envs[i]._row <  self.planning_envs[i]._size:
                    # get them to step to the next state
                    _ = self.planning_envs[i]._step(actions[i])

                # get the observation
                next_observations.append(self.planning_envs[i]._get_observation())

        # stack them in dim 0
        next_observations = np.stack(next_observations, axis=0)

        return next_observations

    def get_next_true_observation_indexes(self, indexes, actions):
        """"
            Expects a indexes numpy array of shape (num_envs, 2) and actions a list of length num_envs
        """
        assert len(np.shape(indexes)) == 2
        assert len(actions) == np.shape(indexes)[0] == self.num_envs

        # Identify the rows and columns for each state
        next_observations_indexes = []
        for i in range(self.num_envs):
            # If this is the null state, can just retain the null state
            if indexes[i][0] >= self.planning_envs[i]._size:
                next_observations_indexes.append(indexes[i])
            else:
                # set the envs to the right state
                self.planning_envs[i]._row = indexes[i][0]
                self.planning_envs[i]._column = indexes[i][1]

                # get them to step to the next state
                _ = self.planning_envs[i]._step(actions[i])

                # get the indexes
                next_observations_indexes.append([self.planning_envs[i]._row, self.planning_envs[i]._column])

        # stack them in dim 0
        next_observations_indexes = np.stack(next_observations_indexes, axis=0)

        return next_observations_indexes

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
        # If the state is just one state in indexes form
        if len(np.shape(state)) == 1:
            if use_state_visits:
                raise NotImplementedError
            else:
                if state[0] >= self.planning_envs[0]._size:
                    return 0
                else:
                    env_state_action_visit_counter = self.sa_counts[state[0], state[1], action]
                    return self.scale / (env_state_action_visit_counter + self.eps)
        # If state is a tensor of shape (num_envs, stacked_obs, height, width)
        if len(np.shape(state)) == 3:
            num_envs = np.shape(state)[0]
            reward_uncertainties = []
            assert len(action) == num_envs    # verify that the number of actions matches the number of states
            for i in range(num_envs):
                env_state = state[i, :, :]
                index_row, index_column = self.from_one_hot_state_to_indexes(env_state)
                # If this state is outside of bounds, return 0 uncertainty
                if index_row < 0 or index_column < 0:
                    reward_uncertainties.append(0)
                else:
                    env_state_action_visit_counter = self.sa_counts[index_row, index_column, action[i]]
                    reward_uncertainties.append(self.scale / (env_state_action_visit_counter + self.eps))
            return reward_uncertainties
        # If state was given as indexes:
        elif np.shape(state) == (self.num_envs, 2):
            num_envs = np.shape(state)[0]
            reward_uncertainties = []
            assert len(action) == num_envs  # verify that the number of actions matches the number of states
            # if we want to use the state visits as proxy for uncertainty:
            if use_state_visits:
                # We need to first compute the next states
                next_states = self.get_next_true_observation_indexes(state, actions=action)
                for i in range(num_envs):
                    [index_row, index_column] = state[i]
                    [index_row_next_state, index_col_next_state] = next_states[i, 0], next_states[i, 1]
                    # If state is already outside bound of env, append 0
                    if index_row >= self.planning_envs[0]._size:
                        reward_uncertainties.append(0)
                    # If the action just took us outside of the bound of env, take the unc. from state-action
                    elif index_row_next_state == self.planning_envs[0]._size:
                        reward_uncertainties.append(self.scale / (self.sa_counts[index_row, index_column, action[i]] + self.eps))
                    # otherwise, just compute the uncertainty according to the state counter
                    else:
                        reward_uncertainties.append(self.scale / (self.s_counts[index_row_next_state, index_col_next_state] + self.eps))
                return reward_uncertainties
            for i in range(num_envs):
                [index_row, index_column] = state[i]
                # If this state is outside of bounds, return 0 uncertainty
                if index_row >= self.planning_envs[0]._size:
                    reward_uncertainties.append(0)
                else:
                    env_state_action_visit_counter = self.sa_counts[index_row, index_column, action[i]]
                    reward_uncertainties.append(self.scale / (env_state_action_visit_counter + self.eps))
            return reward_uncertainties
        elif len(np.shape(state)) == 2:
            index_row, index_column = self.from_one_hot_state_to_indexes(state)
            if index_row < 0 or index_column < 0:
                return 0
            else:
                state_action_visit_counter = self.sa_counts[index_row, index_column, action]
                reward_uncertainty = self.scale / (state_action_visit_counter + self.eps)
                return reward_uncertainty
        else:
            raise ValueError(f"get_reward_uncertainty is not implemented for states of shape "
                             f"!= (num_envs, stacked_obs, height, width) or (height, width), and shape was: {state.shape}")

    def get_surface_value_uncertainty(self, state, use_state_visits=False):
        """
            If state is a tensor of shape (num_envs, stacked_obs, height, width) = (4, 4, 10, 10):
                evaluate the reward-uncertainty for each state and each possible action, take the average and then
                factor the assumption that
            If state is a tensor of shape (height, width):
                evaluate the reward-uncertainty for each action, and return the average
        """
        if len(np.shape(state)) == 3:
            num_envs = np.shape(state)[0]
            # find horizon:
            h = self.observation_space_shape[0]
            # get individual rows from each state:
            current_rows = [self.from_one_hot_state_to_indexes(state[i, :, :])[0] for i in range(num_envs)]
            per_env_horizons = [h - current if current >=0 else -1 for current in current_rows]
            # compute the discount factors for each value-uncertainty
            discount_factors = [(1 - self.gamma ** (2 * horizon)) / (1 - self.gamma ** 2)
                                for horizon in per_env_horizons]
            # compute the uncertainties for each action
            first_actions = [0] * num_envs
            second_actions = [1] * num_envs
            reward_uncertainties_first_action = self.get_reward_uncertainty(state, first_actions, use_state_visits)
            reward_uncertainties_second_action = self.get_reward_uncertainty(state, second_actions, use_state_visits)
            return [abs(discount_factor * (first_unc + second_unc) / 2) for first_unc, second_unc, discount_factor in
                    zip(reward_uncertainties_first_action, reward_uncertainties_second_action, discount_factors)]
        elif np.shape(state) == (self.num_envs, 2):
            num_envs = np.shape(state)[0]
            # find horizon:
            h = self.observation_space_shape[0]
            # get individual rows from each state:
            current_rows = [state[i][0] for i in range(num_envs)]
            per_env_horizons = [h - current for current in current_rows]
            # compute the discount factors for each value-uncertainty
            discount_factors = [(1 - self.gamma ** (2 * horizon)) / (1 - self.gamma ** 2)
                                for horizon in per_env_horizons]
            # compute the uncertainties for each action
            first_actions = [0] * num_envs
            second_actions = [1] * num_envs
            reward_uncertainties_first_action = self.get_reward_uncertainty(state, first_actions, use_state_visits)
            reward_uncertainties_second_action = self.get_reward_uncertainty(state, second_actions, use_state_visits)
            return [abs(discount_factor * (first_unc + second_unc) / 2) for first_unc, second_unc, discount_factor in
                    zip(reward_uncertainties_first_action, reward_uncertainties_second_action, discount_factors)]
        elif len(np.shape(state)) == 2:
            h = self.observation_space_shape[0]
            row = self.from_one_hot_state_to_indexes(state)[0]
            # If this state is outside the bounds of the environment, return 0 uncertainty
            if row < 0:
                return 0
            horizon = h - row
            discount_factor = (1 - self.gamma ** (2 * horizon)) / (1 - self.gamma ** 2)
            first_action_reward_unc = self.get_reward_uncertainty(state, 0, use_state_visits) * discount_factor
            second_action_reward_unc = self.get_reward_uncertainty(state, 1, use_state_visits) * discount_factor
            return (first_action_reward_unc + second_action_reward_unc) / 2
        else:
            raise ValueError(f"get_surface_value_uncertainty is not implemented for states of shape "
                             f"!= (num_envs, height, width) or (height, width), and shape was: {state.shape}")

    def get_propagated_value_uncertainty(self, state, propagation_horizon=3, sampling_times=1, use_state_visits=False):
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
        return propagated_uncertainty.tolist()

    def recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits=False):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0:
            return np.asarray(self.get_surface_value_uncertainty(state))
        else:
            local_reward_uncertainty = (np.asarray(self.get_reward_uncertainty(state, [0 for _ in range(self.num_envs)], use_state_visits)) + np.asarray(self.get_reward_uncertainty(state, [1 for _ in range(self.num_envs)], use_state_visits))) / 2
            first_next_state, second_next_state = self.get_next_true_observation_indexes(state, actions=[0 for _ in range(self.num_envs)]), self.get_next_true_observation_indexes(state, actions=[1 for _ in range(self.num_envs)])
            return np.asarray(local_reward_uncertainty + (self.gamma ** 2) * np.max(self.recursive_propagated_uncertainty(first_next_state, propagation_horizon - 1, use_state_visits), self.recursive_propagated_uncertainty(second_next_state, propagation_horizon - 1, use_state_visits)))

    def sampled_recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0:
            return np.asarray(self.get_surface_value_uncertainty(state))
        else:
            actions = [np.random.randint(low=0,high=2) for _ in range(self.num_envs)]
            local_reward_uncertainty = np.asarray(self.get_reward_uncertainty(state, actions, use_state_visits))
            sampled_next_state = self.get_next_true_observation_indexes(state, actions=actions)
            return np.asarray(local_reward_uncertainty + (self.gamma ** 2) * (self.sampled_recursive_propagated_uncertainty(sampled_next_state, propagation_horizon - 1, use_state_visits)))
