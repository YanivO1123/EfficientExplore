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

    @staticmethod
    def from_one_hot_state_to_indexes(state):
        """
            Takes a numpy array, one-hot encoded state of shape (h, w) and returns the indexes of the 1 in the encoding
            If state is the "null state" (all zeros), returns negative indexes
        """
        assert len(np.shape(state)) == 2
        # If the given state is the "null state" (outside of env bounds)
        if not state.any():
            return -1, -1
        else:
            indexes = (state == 1).nonzero()
            index_row, index_column = indexes[0][0], indexes[1][0]
            return index_row, index_column

    def observe(self, state, action):
        """ Add counts for observed 'state' (observations from deep_sea, which are also the true state).
            'state' is a numpy array of shape (height, width)
        """
        assert len(np.shape(state)) == 2
        # The shape of the expected input is:
        # (height, width)
        row, column = self.from_one_hot_state_to_indexes(state)
        if row < 0 or column < 0:
            raise ValueError(f"Tried to observe the 'null' state. State was: {state}")
        self.s_counts[row, column] += 1
        self.sa_counts[row, column, action] += 1

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

                # get them to step to the next state
                _ = self.planning_envs[i]._step(actions[i])

                # get the observation
                next_observations.append(self.planning_envs[i]._get_observation())

        # stack them in dim 0
        next_observations = np.stack(next_observations, axis=0)

        return next_observations

    def get_reward_uncertainty(self, state, action):
        """
            If state is an array of shape (num_envs, height, width) = (4, 10, 10) and action a list of
            len num_envs:
                We return a list of uncertainties, of length num_envs, and all but the last observation in each stack is
                ignored
            If state is a tensor of shape (height, width) and action is a number:
                returns the uncertainty of the state action pair
            Does not change the counters.
        """
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
                    env_state_action_visit_counter = self.sa_counts[index_row, index_column, action[i]].item()
                    reward_uncertainties.append(self.scale / (env_state_action_visit_counter + self.eps))
            return reward_uncertainties
        elif len(np.shape(state)) == 2:
            index_row, index_column = self.from_one_hot_state_to_indexes(state)
            if index_row < 0 or index_column < 0:
                return 0
            else:
                state_action_visit_counter = self.sa_counts[index_row, index_column, action].item()
                reward_uncertainty = self.scale / (state_action_visit_counter + self.eps)
                return reward_uncertainty
        else:
            raise ValueError(f"get_reward_uncertainty is not implemented for states of shape "
                             f"!= (num_envs, stacked_obs, height, width) or (height, width), and shape was: {state.shape}")

    def get_surface_value_uncertainty(self, state):
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
            reward_uncertainties_first_action = self.get_reward_uncertainty(state, first_actions)
            reward_uncertainties_second_action = self.get_reward_uncertainty(state, second_actions)
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
            first_action_reward_unc = self.get_reward_uncertainty(state, 0) * discount_factor
            second_action_reward_unc = self.get_reward_uncertainty(state, 1) * discount_factor
            return (first_action_reward_unc + second_action_reward_unc) / 2
        else:
            raise ValueError(f"get_surface_value_uncertainty is not implemented for states of shape "
                             f"!= (num_envs, height, width) or (height, width), and shape was: {state.shape}")

    def get_propagated_value_uncertainty(self, state, action, propagation_horizon=5):
        """
            Takes a state, action, environment and horizon.
            Uses the environment to plan and compute the value uncertainty based on a trajectory of reward uncertainties
            returns the environment as it recieved it
        """
        print("Getting propagated visitation count uncertainty for deep sea when the actions are randomized is "
              "non-trivial and not implemented")
        raise NotImplementedError
        # propagated_uncertainty = 0
        # original_row, original_column = np.where(state == 1)
        # print(f"original_row, original_column = {original_row}, {original_column}")
        # self.planning_env._row = original_row
        # self.planning_env._column = original_column
        # for i in range(3):
        #     _, _, _, _ = self.planning_env.step(0)
        # obs = self.planning_env._get_observation()
        # print(f"Current state is: np.where(obs == 1) = {np.where(obs == 1)}")
        # self.planning_env._row = original_row
        # self.planning_env._column = original_column
        # obs, _, _, _ = self.planning_env.step(0)
        # obs = self.planning_env._get_observation()
        # print(f"And after resetting the env and taking one more step left: np.where(obs == 1) = {np.where(obs == 1)}")
        # exit()
        #
        # # Add local reward uncertainty
        # propagated_uncertainty += self.get_reward_uncertainty(state, action)
        #
        # for i in range(propagation_horizon):
        #     state = self.planning_env.step(action)
        #     propagated_uncertainty += self.get_reward_uncertainty(state, action)
        #
        # env._row = original_row
        # env._column = original_column
        # return propagated_uncertainty

