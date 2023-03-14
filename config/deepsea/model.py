import math

import numpy
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize

from config.deepsea.extended_deep_sea import DeepSea


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


def no_batch_norm_mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.ReLU,
        init_zero=False,
):
    """MLP layers without batch normalization.
        Parameters
        ----------
        input_size: int
            dim of inputs
        layer_sizes: list
            dim of hidden layers
        output_size: int
            dim of outputs
        init_zero: bool
            zero initialization for the last layer (including w and b).
            This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return torch.nn.Sequential(*layers)


class FullyConnectedEfficientExploreNet(BaseNet):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 fc_state_prediction_layers,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 fc_rnd_layers,
                 fc_rnd_target_layers,
                 fc_ube_layers,
                 reward_support_size,
                 value_support_size,
                 inverse_value_transform,
                 inverse_reward_transform,
                 lstm_hidden_size,
                 momentum=0.1,
                 proj_hid=256,
                 proj_out=256,
                 pred_hid=64,
                 pred_out=256,
                 init_zero=False,
                 rnd_scale=1.0,
                 learned_model=True,
                 env_size=10,
                 mapping_seed=0,
                 randomize_actions=True,
                 uncertainty_type='rnd',
                 discount=0.997,
                 ensemble_size=5,
                 use_prior=True,
                 prior_scale=10,
                 ube_scale=10,
                 ):
        """
            FullyConnected (more precisely non-resnet) EfficientZero network.
            Parameters
            __________
            observation_shape: tuple or list
                shape of observations: [C, W, H] = [1, N, N], for deep sea for which this arch. is implemented.
            learned_model:
            env_size:
            mapping_seed:
            randomize_actions:
        """
        super(FullyConnectedEfficientExploreNet, self).__init__(inverse_value_transform, inverse_reward_transform,
                                                                lstm_hidden_size)
        self.init_zero = init_zero
        self.action_space_size = action_space_size
        # For consistency loss
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        # Planning dynamics with true model or learned model
        self.learned_model = learned_model
        # Uncertainty params
        self.uncertainty_type = uncertainty_type
        self.ensemble_size = ensemble_size
        self.use_prior = use_prior
        self.prior_scale = prior_scale

        # The size of flattened encoded state:
        self.encoded_state_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
        # The size of the input to the dynamics network is (num_channels + the action channel) * H * W
        self.dynamics_input_size = (observation_shape[0] + 1) * observation_shape[1] * observation_shape[2]

        # In this arch. the representation is the original observation
        self.representation_network = torch.nn.Identity()
        self.policy_network = mlp(self.encoded_state_size, fc_policy_layers, action_space_size, init_zero=init_zero,
                                  momentum=momentum)

        if 'ensemble' in uncertainty_type:
            self.value_network = nn.ModuleList([mlp(self.encoded_state_size, fc_value_layers, value_support_size,
                                                    init_zero=init_zero, momentum=momentum)
                                                for _ in range(ensemble_size)])
            if self.use_prior:
                self.value_network_prior = nn.ModuleList([mlp(self.encoded_state_size, fc_value_layers,
                                                              value_support_size, init_zero=init_zero,
                                                              momentum=momentum)
                                                          for _ in range(ensemble_size)])
        else:
            self.value_network = mlp(self.encoded_state_size, fc_value_layers, value_support_size, init_zero=init_zero,
                                     momentum=momentum)

        self.dynamics_network = FullyConnectedDynamicsNetwork(
            env_size,
            mapping_seed,
            randomize_actions,
            self.dynamics_input_size,
            self.encoded_state_size,
            observation_shape,
            fc_state_prediction_layers,
            fc_reward_layers,
            reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=momentum,
            init_zero=init_zero,
            learned_model=learned_model,
            ensemble=True if 'ensemble' in uncertainty_type else False,
            ensemble_size=self.ensemble_size,
            use_prior=self.use_prior,
            prior_scale=self.prior_scale
        )

        # projection
        num_channels = 1
        in_dim = num_channels * observation_shape[1] * observation_shape[2]
        self.porjection_in_dim = in_dim
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

        # RND
        if 'rnd' in uncertainty_type:
            self.rnd_scale = rnd_scale
            self.input_size_value_rnd = self.encoded_state_size
            self.input_size_reward_rnd = observation_shape[0] * observation_shape[1] * observation_shape[2] + \
                                         observation_shape[0] * action_space_size
            # It's important that the RND nets are NOT initiated with zero
            self.reward_rnd_network = no_batch_norm_mlp(self.input_size_reward_rnd, fc_rnd_layers[:-1],
                                                        fc_rnd_layers[-1], init_zero=False)
            self.reward_rnd_target_network = no_batch_norm_mlp(self.input_size_reward_rnd, fc_rnd_target_layers[:-1],
                                                               fc_rnd_target_layers[-1], init_zero=False)
            self.value_rnd_network = no_batch_norm_mlp(self.input_size_value_rnd, fc_rnd_layers[:-1], fc_rnd_layers[-1],
                                                       init_zero=False)
            self.value_rnd_target_network = no_batch_norm_mlp(self.input_size_value_rnd, fc_rnd_target_layers[:-1],
                                                              fc_rnd_target_layers[-1],
                                                              init_zero=False)
            # The value_rnd_unc_prop coeff is the sum of a geometric series with r = gamma ** 2 and n = env_size
            self.value_rnd_propagation_scale = (1 - discount ** (observation_shape[-1] * 2)) / (1 - discount ** 2)

        if 'ube' in uncertainty_type:
            self.ube_scale = ube_scale
            self.ube_network = mlp(self.encoded_state_size, fc_ube_layers, 1,
                                   init_zero=init_zero, momentum=momentum)

    def representation(self, observation):
        # Regardless of the number of stacked observations, we only pass the last
        # observation = observation[:, -1, :, :].unsqueeze(1)
        # Should be faster and doing the same:
        observation = observation[:, -1:, :, :]

        # With the fully-connected deep_sea architecture, we maintain the original observation as the representation
        encoded_state = self.representation_network(observation)
        return encoded_state

    def prediction(self, encoded_state):
        # We reshape the encoded_state to the shape of input of the FC nets that follow
        encoded_state = encoded_state.reshape(-1, self.encoded_state_size)
        policy = self.policy_network(encoded_state)
        if 'ensemble' in self.uncertainty_type:
            if self.use_prior:
                value = [value_net(encoded_state) + self.prior_scale * prior_net(encoded_state.detach()).detach()
                         for value_net, prior_net in zip(self.value_network, self.value_network_prior)]
            else:
                value = [value_net(encoded_state) for value_net in self.value_network]
        else:
            value = self.value_network(encoded_state)

        return policy, value

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],  # batch dimension
                    1,  # channels dimension
                    encoded_state.shape[2],  # H dim
                    encoded_state.shape[3],  # W dim
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
                action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        if self.learned_model:
            next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)
        else:
            next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden, encoded_state,
                                                                                    action)
        return next_encoded_state, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = 0
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.reshape(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

    def compute_value_rnd_uncertainty(self, state):
        state = state.reshape(-1, self.input_size_value_rnd).detach()
        return self.rnd_scale * torch.nn.functional.mse_loss(self.value_rnd_network(state),
                                                             self.value_rnd_target_network(state).detach(),
                                                             reduction='none').sum(dim=-1)

    def compute_reward_rnd_uncertainty(self, state, action):
        # Turn in a state_action vector
        action_one_hot = (
            torch.zeros(
                (
                    state.shape[0],  # batch dimension
                    self.action_space_size
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot[action.long()] = 1
        flattened_state = state.reshape(-1, self.input_size_reward_rnd - self.action_space_size)
        state_action = torch.cat((flattened_state, action_one_hot), dim=1).detach()
        # Compute the RND uncertainty
        return self.rnd_scale * torch.nn.functional.mse_loss(self.reward_rnd_network(state_action),
                                                             self.reward_rnd_target_network(state_action).detach(),
                                                             reduction='none').sum(dim=-1)

    def compute_ube_uncertainty(self, state):
        """
            Returns the value-uncertainty prediction from the UBE network (See UBE paper
            https://arxiv.org/pdf/1709.05380.pdf). The state is always detached, because we don't want the UBE
            prediction to train the learned dynamics and representation networks.
        """
        state = state.reshape(-1, self.encoded_state_size).detach()
        # We squeeze the result to return tensor of shape [B] instead of [B, 1]
        ube_prediction = self.ube_network(state).squeeze()

        if torch.isinf(ube_prediction).any() or torch.isnan(ube_prediction).any():
            print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
                  f"Inf (/nan) in UBE prediction! \n"
                  f"ube_prediction = {ube_prediction} \n"
                  f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
        # To guarantee that output value is positive, we treat it as a logit instead of as a direct scalar
        ube_prediction = torch.exp(ube_prediction)
        if torch.isinf(ube_prediction).any() or torch.isnan(ube_prediction).any():
            print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
                  f"Inf (/nan) in UBE prediction! \n"
                  f"ube_prediction = {ube_prediction} \n"
                  f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")

        return ube_prediction

    def ensemble_prediction_to_variance(self, logits):
        if not isinstance(logits, list):
            return None
        # logits is a list of length ensemble_size, of tensors of shape: (num_parallel_envs, full_support_size)
        with torch.no_grad():
            # Softmax the logits
            logits = [torch.softmax(logits[i], dim=1) for i in range(len(logits))]

            # Stack into a tensor to compute the variance using torch.
            stacked_tensor = torch.stack(logits, dim=0)
            # Shape of stacked_tensor: (ensemble_size, num_parallel_envs, full_support_size)

            # Compute the per-entry variance over the dimension 0 (ensemble size)
            scalar_variance = torch.var(stacked_tensor, unbiased=False, dim=0)
            # Resulting shape of scalar_variance: (num_parallel_envs, full_support_size)

            # Sum the per-entry variance scores
            scalar_variance = scalar_variance.sum(-1)
            # Resulting shape of scalar_variance: (num_parallel_envs)

            return scalar_variance


class FullyConnectedDynamicsNetwork(nn.Module):
    def __init__(self,
                 env_size,
                 mapping_seed,
                 randomize_actions,
                 dynamics_input_size,
                 hidden_state_size,
                 hidden_state_shape,
                 fc_state_prediction_layers,
                 fc_reward_layers,
                 full_support_size,
                 lstm_hidden_size=64,
                 momentum=0.1,
                 init_zero=False,
                 learned_model=True,
                 ensemble=False,
                 ensemble_size=5,
                 use_prior=True,
                 prior_scale=10,
                 ):
        """
        Non-resnet, non-conv dynamics network, for deep_sea.
        Implements dynamics planning with the true model of deep_sea, if learned_model=False
        Parameters
        __________
        dynamics_input_size: int
            The size of the input of hidden_state + onehot action encoding
        hidden_state_size: int
            The size of the hidden state flattened for FC
        hidden_state_shape: tuple or list
            The shape of the hidden state is expected to be in, without the batch dim, [C, H, W] = [1, N, N]
        """
        super().__init__()
        self.hidden_state_shape = hidden_state_shape
        self.dynamics_input_size = dynamics_input_size
        # True-model planning params
        self.env_size = env_size
        self.learned_model = learned_model
        # Ensemble params
        self.ensemble_size = ensemble_size
        self.ensemble = ensemble
        self.use_prior = use_prior
        self.prior_scale = prior_scale

        # Init dynamics prediction, learned or given
        if learned_model:
            self.state_prediction_net = mlp(dynamics_input_size, fc_state_prediction_layers, hidden_state_size,
                                            init_zero=init_zero,
                                            momentum=momentum)
            self.action_mapping = None
        else:
            self.action_mapping = torch.from_numpy(DeepSea(size=env_size, mapping_seed=mapping_seed, seed=mapping_seed,
                                                           randomize_actions=randomize_actions)._action_mapping).long()
            self.action_mapping = self.action_mapping * 2 - 1

        # The input to the lstm is the concat tensor of hidden_state and action
        self.lstm = nn.LSTM(input_size=dynamics_input_size, hidden_size=lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(lstm_hidden_size, momentum=momentum)
        if self.ensemble:
            self.fc = nn.ModuleList([mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                                         momentum=momentum)
                                     for _ in range(ensemble_size)])
            if self.use_prior:
                self.fc_net_prior = nn.ModuleList([mlp(lstm_hidden_size, fc_reward_layers, full_support_size,
                                                       init_zero=init_zero, momentum=momentum)
                                                   for _ in range(ensemble_size)])
        else:
            self.fc = mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                          momentum=momentum)

    def forward(self, x, reward_hidden, current_state=None, action=None):
        # Flatten input state-action for FC nets
        x = x.view(-1, self.dynamics_input_size)

        if self.learned_model:
            # Next-state prediction is done based on a FC network
            next_state = self.state_prediction_net(x)
            next_state = nn.functional.relu(next_state)
            # Reshape the state to the shape MuZero expects: [num_envs or batch_size, channels, H, W] = [B, 1, N, N]
            next_state = next_state.view(-1,
                                         self.hidden_state_shape[0],
                                         self.hidden_state_shape[1],
                                         self.hidden_state_shape[2])
        else:
            assert current_state is not None and action is not None, f"Cannot plan with true deep_sea model in " \
                                                                     f"dynamics without current_state and action inputs"
            # Produce next states from previous state and action, in the shape MuZero expects:
            # [num_envs or batch_size, channels, H, W] = [B, 1, N, N]
            next_state = self.get_batched_next_states(current_state, action)

        # Reward prediction is done based on EfficientExplore architecture
        x = x.unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        if self.ensemble:
            if self.use_prior:
                value_prefix = [value_prefix_net(value_prefix) + self.prior_scale *
                                prior_net(value_prefix.detach()).detach() for value_prefix_net, prior_net
                                in zip(self.fc, self.fc_net_prior)]
            else:
                value_prefix = [value_prefix_net(value_prefix) for value_prefix_net in self.fc]
        else:
            value_prefix = self.fc(value_prefix)

        return next_state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        if self.learned_model:
            dynamic_mean = []
            for name, param in self.state_prediction_net.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
            dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        else:
            dynamic_mean = 0
        return dynamic_mean

    def get_reward_mean(self):
        for index, (name, param) in enumerate(self.fc.named_parameters()):
            if index > 0:
                temp_weights = param.detach().cpu().numpy().reshape(-1)
                reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
            else:
                reward_w_dist = param.detach().cpu().numpy().reshape(-1)
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean

    def get_batched_next_states(self, batched_states, batched_actions):
        """
        Shape of batched_states is (B, H, W)
        Shape of batched_actions is (B, 1)
        """
        batch_size = batched_states.shape[0]
        N = batched_states.shape[-1]

        # Flatten 1 hot representation
        flattened_batched_states = batched_states.reshape(shape=(batch_size, N * N)).to(batched_states.device)

        # Get the indexes of the 1 hot along the B dimension, and shape them to 1 dim vectors of size batch_size
        rows, columns = (flattened_batched_states.argmax(dim=1) // N).long().squeeze().to(batched_states.device), (
                flattened_batched_states.argmax(dim=1) % N).long().squeeze().to(batched_states.device)

        # Switch actions to -1, +1
        batched_actions = batched_actions.squeeze() * 2 - 1

        # action_mapping expected to already be in the form -1, +1, this line is here for logic
        # self.action_mapping = self.action_mapping * 2 - 1

        # Expand the action mapping
        action_mapping = self.action_mapping.unsqueeze(0).expand(batch_size, -1, -1).to(batched_states.device)

        # Flatten to shape [B, N * N]
        flattened_action_mapping = action_mapping.reshape(batch_size, N * N).to(batched_states.device)

        # Get all the actions_right with gather in the flattened (row, col) indexes.
        flattened_indexes = (rows * N + columns).to(batched_states.device)
        # actions_right should be of shape [B]
        actions_right = flattened_action_mapping.take(flattened_indexes).to(batched_states.device)

        # All states go down 1 row
        rows = rows + 1

        # Change the actions right from which action is right to what needs to happen to column.
        # This is done by -1 * -1 = move one to the right, -1 * 1 = move 1 to the left
        change_in_column = (actions_right * batched_actions).to(batched_states.device)

        # Change the column values
        columns = torch.clip(columns + change_in_column, min=0).to(batched_states.device)

        # 1-hot the rows and column back into a tensor for shape [B, (N + 1) x (N + 1)]
        flattened_batched_next_states = torch.zeros(size=(batch_size, (N + 1) * (N + 1))) \
            .to(batched_states.device) \
            .scatter_(dim=-1, index=rows.unsqueeze(-1) * (N + 1) + columns.unsqueeze(-1), value=1)

        # Reshape tensor back to B, 1, (N + 1), (N + 1)
        batched_next_states = flattened_batched_next_states.reshape(shape=(batch_size, N + 1, N + 1)).unsqueeze(1)

        # Return the top "corner" of size N x N
        batched_next_states = batched_next_states[:, :, :-1, :-1]

        return batched_next_states
