import numpy as np
import torch
from bsuite import sweep
# from bsuite.environments.deep_sea import DeepSea
# from config.deepsea.extended_deep_sea import DeepSea
from additional_envs.extended_deep_sea import DeepSea
import traceback

# Taken from Wendelin's implementation in the DRL HW explore
class CountUncertainty:
    """
        Defines an uncertainty estimate based on counts over the state/observation space.
        Uncertainty will be scaled by 'scale'.
        Only implemented for the deep_sea environment
    """
    def __init__(self, name, num_envs, mapping_seed, scale=1, epsilon=0.01, fake=False, randomize_actions=True):
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
        self.planning_env = DeepSea(size=self.size, mapping_seed=mapping_seed, seed=mapping_seed, randomize_actions=randomize_actions)
        self.observation_counter = 0
        self.rewarding_transition = self.identify_rewarding_transition()
        self.fake_reward_uncertainty = fake
        self.action_mapping = self.planning_env._action_mapping * 2 - 1

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
        if len(np.shape(state)) == 2:   # If state is of shape [N, N]
            # If the given state is the "null state" (outside of env bounds)
            if not state.any():
                return -1, -1
            else:
                indexes = (state == 1).nonzero()
                index_row, index_column = indexes[0][0], indexes[1][0]
                return index_row, index_column
        elif len(np.shape(state)) == 3:     # If state is of shape [B, N, N]
            # Flatten 1 hot representation
            state = np.asarray(state)
            batch_size = np.shape(state)[0]
            N = np.shape(state)[-1]
            # Flatten 1 hot representation
            flattened_batched_state = state.reshape((batch_size, N * N))
            # Get the indexes of the 1 hot along the B dimension, and shape them to 1 dim vectors of size batch_size
            rows, columns = (flattened_batched_state.argmax(axis=1) // N).astype(dtype=int).reshape(batch_size), (
                    flattened_batched_state.argmax(axis=1) % N).astype(dtype=int).reshape(batch_size)

            indexes_of_states = np.stack((rows, columns), axis=1)

            # Unvectorized code
            # indexes_of_states = []
            # for i in range(batch_size):
            #     if not state[i].any():
            #         # If the given state is the "null state" (outside of env bounds)
            #         indexes_of_states.append([-1, -1])
            #     else:
            #         indexes = (state[i] == 1).nonzero()
            #         index_row, index_column = indexes[0][0], indexes[1][0]
            #         indexes_of_states.append([index_row, index_column])
            return indexes_of_states
        elif len(np.shape(state)) == 4:
            # Flatten 1 hot representation
            state = np.asarray(state)
            batch_size = state.shape[0]
            unroll_length = state.shape[1]
            N = state.shape[-1]
            # Flatten 1 hot representation
            flattened_batched_state = state.reshape((batch_size, unroll_length, N * N))
            # Get the indexes of the 1 hot along the B dimension, and shape them to 1 dim vectors of size batch_size
            rows, columns = (flattened_batched_state.argmax(axis=-1) // N).astype(dtype=int).reshape((batch_size, unroll_length)), \
                            (flattened_batched_state.argmax(axis=-1) % N).astype(dtype=int).reshape((batch_size, unroll_length))

            indexes_of_states = np.stack((rows, columns), axis=1)
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
            # If the state is not the null state
            if row >= 0 and column >= 0:
                self.s_counts[row, column] += 1
                self.sa_counts[row, column, action] += 1
                self.observation_counter += 1
        # Alternatively, if we got a state in indexes form
        elif len(state) == 2 and not len(np.shape(state)) == 2:
            assert state[0] < self.size and state[0] < self.size
            self.s_counts[state[0], state[1]] += 1
            self.sa_counts[state[0], state[1], action] += 1
            self.observation_counter += 1
        # Alternatively, if we got a state in [B, H, W] form
        elif len(state.shape) == 3:
            batch_size, H, W = state.shape
            action = action.reshape(batch_size)
            indexes_of_states = self.from_one_hot_state_to_indexes(state)
            rows, columns = indexes_of_states[:, 0], indexes_of_states[:, 1]
            for i in range(batch_size):
                row, column = rows[i], columns[i]
                if row >= 0 and column >= 0:
                    self.s_counts[row, column] += 1
                    self.sa_counts[row, column, action[i]] += 1
                    self.observation_counter += 1
        elif len(state.shape) == 4:
            batch_size, unroll_length, H, W = state.shape
            action = action.reshape(batch_size, unroll_length)
            batch_size = state.shape[0]
            unroll_length = state.shape[1]
            rows, columns = self.from_one_hot_state_to_indexes(state)
            assert state.shape[0] == action.shape[0] and state.shape[1] == action.shape[1], \
                f"state.shape = {state.shape} and action.shape = {action.shape}, and should be equal across first two " \
                f"dims (batch size and unroll length)"
            for i in range(batch_size):
                for j in range(unroll_length):
                    row, column = rows[i, j], columns[i, j]
                    # If the state is not the null state
                    if row >= 0 and column >= 0:
                        self.s_counts[row, column] += 1
                        self.sa_counts[row, column, action[i, j]] += 1
                        self.observation_counter += 1
        else:
            raise ValueError(f"Received state in unexpected form. "
                             f"Can only observe states of shapes [B, H, W], [H, W] and [2], "
                             f"and received state of shape: {state.shape} and action of shape {action.shape}")

    def get_next_true_observation(self, states, actions):
        """"
            Expects a states numpy array of shape (num_envs, w, h) and actions a list of length num_envs
        """
        assert len(np.shape(states)) == 3
        assert len(actions) == np.shape(states)[0]

        batch_size = np.shape(states)[0]

        # Identify the rows and columns for each state
        indexes = [(self.from_one_hot_state_to_indexes(states[i])) for i in range(batch_size)]
        next_observations = []
        for i in range(batch_size):
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
            assert len(actions) == np.shape(indexes)[0]

            # Vectorized code is somehow slower. Using non-vectorized code instead

            # # Vectorized code
            # batch_size = np.shape(indexes)[0]
            # N = self.size
            #
            # indexes = np.asarray(indexes)
            #
            # # Switch actions to -1, +1
            # batched_actions = np.asarray(actions).squeeze() * 2 - 1
            #
            # # action_mapping expected to already be in the form -1, +1, this line is here for logic
            # # self.action_mapping = self.planning_env._action_mapping * 2 - 1
            #
            # # Expand the action mapping
            # action_mapping = np.expand_dims(self.action_mapping, axis=0)
            # action_mapping = np.tile(action_mapping, (batch_size, 1, 1))
            #
            # # Flatten to shape [B, N * N]
            # flattened_action_mapping = action_mapping.reshape(batch_size, N * N)
            #
            # rows, columns = indexes[:, 0], indexes[:, 1]
            #
            # # Get all the actions_right with gather in the flattened (row, col) indexes.
            # flattened_indexes = rows * N + columns
            # # actions_right should be of shape [B]
            # actions_right = flattened_action_mapping.take(flattened_indexes)
            #
            # # All states go down 1 row
            # rows = rows + 1
            #
            # # Change the actions right from which action is right to what needs to happen to column.
            # # This is done by -1 * -1 = move one to the right, -1 * 1 = move 1 to the left
            # change_in_column = actions_right * batched_actions
            #
            # # Change the column values
            # columns = np.clip(columns + change_in_column, a_min=0, a_max=None)
            #
            # # 1-hot the rows and column back into a tensor for shape [B, (N + 1) x (N + 1)]
            # flattened_batched_next_states = np.zeros((batch_size, (N + 1) * (N + 1)))
            # indexes = np.expand_dims(rows, axis=1) * (N + 1) + np.expand_dims(columns, axis=1)
            # np.put_along_axis(flattened_batched_next_states, indices=indexes, values=1, axis=1)
            #
            # # Reshape tensor back to B, 1, (N + 1), (N + 1)
            # batched_next_states = flattened_batched_next_states.reshape((batch_size, N + 1, N + 1))
            # # Return the top "corner" of size N x N
            # batched_next_states = batched_next_states[:, :-1, :-1]
            #
            # # Transform back to indexes
            # next_observations_indexes = self.from_one_hot_state_to_indexes(batched_next_states)
            #
            # # return
            # return next_observations_indexes

            # Unvectorized code
            batch_size = np.shape(indexes)[0]
            # Identify the rows and columns for each state
            next_observations_indexes = []
            for i in range(batch_size):
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

    def state_action_uncertainty(self, index_row, index_column, action, use_state_visits=False):
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
            If state is an array of shape (num_envs, height, width) and action a list of
            len num_envs:
                We return a list of uncertainties, of length num_envs, and all but the last observation in each stack is
                ignored
            If state is a tensor of shape (height, width) and action is a number:
                returns the uncertainty of the state action pair
            Does not change the counters.
        """
        # If state is a tensor of shape [B, S x C, H, W]:
        if len(np.shape(state)) == 4:
            state = np.asarray(state)
            # First we need to rework this to shape [B, -1, H, W]
            state = state[:, -1, :, :]
            # Make sure the new shape is [B, H, W]
            # state = state.squeeze()
            # Then we need to get the indexes
            state = self.from_one_hot_state_to_indexes(state)
            # Finally, compute the uncertainty for each state
            reward_uncertainties = []
            for i in range(len(state)):
                row, column = state[i]
                reward_uncertainty = self.state_action_uncertainty(row, column, action[i], use_state_visits)
                reward_uncertainties.append(reward_uncertainty)
            return np.asarray(reward_uncertainties)
        # If the state is just one state in indexes form [row, col]
        elif len(np.shape(state)) == 1:
            return self.state_action_uncertainty(state[0], state[1], action, use_state_visits)
        elif len(np.shape(state)) == 2:
            reward_uncertainties = []
            assert len(action) == np.shape(state)[0]  # verify that the number of actions matches the number of states
            batch_size = np.shape(state)[0]
            for i in range(batch_size):
                reward_uncertainties.append(self.state_action_uncertainty(state[i][0], state[i][1], action[i],
                                                                          use_state_visits=use_state_visits))
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
        if len(np.shape(state)) == 2:
            batch_size = np.shape(state)[0]
            # find horizon:
            h = self.size
            # get individual rows from each state:
            current_rows = [state[i][0] for i in range(batch_size)]
            # The horizons are size - row if row is in env, otherwise 0
            per_env_horizons = [h - current if current > 0 else 0 for current in current_rows]
            # compute the discount factors for each value-uncertainty
            # discount_factors = np.asarray([(1 - self.gamma ** (2 * horizon)) / (1 - self.gamma ** 2)
            #                     for horizon in per_env_horizons])
            # compute the uncertainties for each action
            first_actions = [0] * batch_size
            second_actions = [1] * batch_size
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
        if len(np.shape(state)) == 4:   # If shape = [B, C * S, H, W]
            state = np.asarray(state)
            # First we need to rework this to shape [B, 1, H, W] by taking the last observation in S * C
            # Because C = 1 and S is stacked obs.
            state = state[:, -1, :, :]
            # Make sure the new shape is [B, H, W]
            # state = state.squeeze()
            # Then we need to get the indexes
            state = self.from_one_hot_state_to_indexes(state)
            # Finally, we can compute the unc.
            if sampling_times > 0:
                samples = []
                for i in range(sampling_times):
                    propagated_uncertainty = self.sampled_recursive_propagated_uncertainty(state, propagation_horizon, use_state_visits)
                    samples.append(propagated_uncertainty)
                propagated_uncertainty = np.stack(samples, axis=0).max(axis=0)
            else:
                propagated_uncertainty = self.recursive_propagated_uncertainty(np.asarray(state), propagation_horizon, use_state_visits)
        elif len(np.shape(state)) == 2:
            if sampling_times > 0:
                samples = []
                for i in range(sampling_times):
                    propagated_uncertainty = self.sampled_recursive_propagated_uncertainty(np.asarray(state), propagation_horizon, use_state_visits)
                    samples.append(propagated_uncertainty)
                propagated_uncertainty = np.stack(samples, axis=0).max(axis=0)
            else:
                propagated_uncertainty = self.recursive_propagated_uncertainty(np.asarray(state), propagation_horizon, use_state_visits)
        else:
            raise ValueError(f"get_propagated_value_uncertainty not implemented for received state shape. "
                             f"get_propagated_value_uncertainty is implemented for len(np.shape(state)) = 4 or = 2, "
                             f"but np.shape(state) was {np.shape(state)}")
        return propagated_uncertainty

    def recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits=False):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0 or min(state[:, 0]) >= self.observation_space_shape[0] - 1:
            return np.asarray(self.get_surface_value_uncertainty(state, use_state_visits))
        else:
            batch_size = np.shape(state)[0]
            left_reward_uncertainty = self.get_reward_uncertainty(state, [0 for _ in range(batch_size)], use_state_visits)
            right_reward_uncertainty = self.get_reward_uncertainty(state, [1 for _ in range(batch_size)], use_state_visits)
            left_next_state = self.get_next_true_observation_indexes(state, actions=[0 for _ in range(batch_size)])
            right_next_state = self.get_next_true_observation_indexes(state, actions=[1 for _ in range(batch_size)])
            return np.maximum(
                left_reward_uncertainty + self.recursive_propagated_uncertainty(left_next_state,
                                                                                propagation_horizon - 1,
                                                                                use_state_visits),
                right_reward_uncertainty + self.recursive_propagated_uncertainty(right_next_state,
                                                                                 propagation_horizon - 1,
                                                                                 use_state_visits)
            )

    def sampled_recursive_propagated_uncertainty(self, state, propagation_horizon, use_state_visits):
        """
            Operates on arrays
            Takes state of shape (num_envs, 2) and averages the local reward uncertainties and surface value uncertainty
        """
        if propagation_horizon == 0 or min(state[:, 0]) >= self.observation_space_shape[0]:
            return np.asarray(self.get_surface_value_uncertainty(state, use_state_visits))
        else:
            batch_size = np.shape(state)[0]
            actions = [np.random.randint(low=0, high=2) for _ in range(batch_size)]
            local_reward_uncertainty = self.get_reward_uncertainty(state, actions, use_state_visits)
            sampled_next_state = self.get_next_true_observation_indexes(state, actions=actions)
            return local_reward_uncertainty + self.sampled_recursive_propagated_uncertainty(sampled_next_state,
                                                                                            propagation_horizon - 1,
                                                                                            use_state_visits)
