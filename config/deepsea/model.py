import math

import numpy
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet

from config.deepsea.extended_deep_sea import DeepSea
import itertools


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
        bias=True,
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
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias), act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        if bias:
            layers[-2].bias.data.fill_(0)

    return torch.nn.Sequential(*layers)


class FullyConnectedEfficientExploreNet(BaseNet):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 representation_type,
                 fc_representation_layers,
                 fc_state_prediction_layers,
                 fc_state_prediction_prior_layers,
                 fc_reward_layers,
                 fc_reward_prior_layers,
                 fc_value_layers,
                 fc_value_prior_layers,
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
                 mapping_seed=0,
                 randomize_actions=True,
                 uncertainty_type='ensemble_ube',
                 discount=0.997,
                 ensemble_size=5,
                 use_prior=True,
                 prior_scale=10.0,
                 encoder_layers=None,
                 encoding_size=0,
                 categorical_ube=False,
                 inverse_ube_transform=None,
                 ube_support_size=None
                 ):
        """
            FullyConnected (more precisely non-resnet) EfficientZero network. Based on the architecture of
            EfficientZeroNet from config.atari.model.
            Additional-to-deep_sea parameters are specified below.
            (Additional) Parameters
            __________
            observation_shape: tuple or list
                shape of observations: [C, W, H] = [1, N, N], for deep sea for which this arch. is implemented.
            representation_type: string
                options: [learned, identity, concatted, encoder]
                meanings:
                    learned: standard MuZero learned representation
                    identity: the representation function is the identity function
                    concatted: the representation function takes 1 hot state of shape (B, 1, N, N) and returns a state
                        of shape (B, 2 * N), made of of a row one-hot vector concatted with a column one-hot vector.
                    encoder: the representation function is a randomly-initialized, untrained encoder, translating
                        the state space from (N, N) to (encoding_size, encoding_size)
            learned_model: bool
                Whether the transition model is learned (true, MuZero), or given (False, ZetaZero, can be thought of as
                AlphaZero with learned reward function)
            mapping_seed: int
                The seed initializing the random actions of deep_sea, for planning with the true transition model.
            randomize_actions: bool
                Whether actions in deep_sea are randomized or not, when planning with a true model.
            uncertainty_type: string
                Specifies the type of uncertainty architecture used. Options: 'ensemble', 'ensemble_ube', 'rnd',
                'rnd_ube'.
            discount: float
                Used to compute the value-uncertainty-propagation scale. This scale guarantees that unseen states
                (high uncertainty from rnd or ensemble, but low from UBE), communicate their uncertainty at the right
                scale.
            ensemble_size: int
                The size of the ensemble.
            use_prior: bool
                Whether to use network-prior (See http://128.84.4.34/abs/1806.03335) on the ensemble members, or not.
            prior_scale: float
                The scale of the contribution of the prior function to the computation.
            use_encoder: bool
                Whether to use a random encoder or not, to reduce the size of the state space.
            encoder_layers: list[int]
                The architecture of the randomly initialized, untrained, (non-BN) MLP that is the encoder.
            encoding_size: int
                The sqrt of the size of the encoded state, so that the structure (C * S, H, W) can be maintained.
                Only used with the encoder.
        """
        super(FullyConnectedEfficientExploreNet, self).__init__(inverse_value_transform, inverse_reward_transform,
                                                                lstm_hidden_size)
        self.init_zero = init_zero
        self.action_space_size = action_space_size
        # For Efficient-zero style consistency loss
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
        self.env_size = observation_shape[-1]
        self.representation_type = representation_type
        self.use_encoder = 'encoder' in self.representation_type
        self.use_identity_representation = 'identity' in self.representation_type
        # The value_uncertainty_propagation_scale coeff is the sum of a geometric series with r = gamma ** 2 and
        # n = env_size. The idea is that for new states local variance prediction should be more reliable than ube
        self.value_uncertainty_propagation_scale = (1 - discount ** (observation_shape[-1] * 2)) / (1 - discount ** 2)
        self.amplify_one_hot = 10

        assert 'identity' in self.representation_type or 'encoder' in self.representation_type \
               or 'concatted' in self.representation_type or 'learned' in self.representation_type, \
            f" self.representation_type = {self.representation_type} and should be one of the following: " \
            f"'identity' 'encoder' 'concatted' 'learned'"

        if self.use_encoder:
            assert encoder_layers is not None, f"encoder_layers must be a list of ints, and was {encoder_layers}"
            assert encoding_size > 0, f"encoding_size must be an int greater than zero, and was {encoding_size}"
            self.encoding_size = encoding_size
            self.encoded_state_size = encoding_size * encoding_size
            self.dynamics_input_size = self.encoded_state_size + action_space_size
            encoder_input_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
            self.representation_encoder = no_batch_norm_mlp(encoder_input_size, encoder_layers, self.encoded_state_size,
                                                            init_zero=False)
            hidden_state_shape = (1, encoding_size, encoding_size)  # C * S, W, H
        # In this arch. the representation is the original observation
        elif self.use_identity_representation:
            self.representation_network = torch.nn.Identity()
            # The size of flattened encoded state:
            self.encoded_state_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
            # The size of the input to the dynamics network is C * S * H * W + C * S * action_space
            self.dynamics_input_size = observation_shape[0] * observation_shape[1] * observation_shape[2] + \
                                       observation_shape[0] * action_space_size
            hidden_state_shape = observation_shape
        elif 'learned' in self.representation_type:
            assert fc_representation_layers is not None and encoding_size >= 1, \
                f"Requires: fc_representation_layers not None, encoding_size >= 1. \n" \
                f"Got: fc_representation_layers = {fc_representation_layers}, encoding_size = {encoding_size}"
            self.encoding_size = encoding_size
            self.encoded_state_size = encoding_size * encoding_size
            self.dynamics_input_size = self.encoded_state_size + action_space_size
            representation_input_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
            hidden_state_shape = (1, encoding_size, encoding_size)
            self.representation_network = no_batch_norm_mlp(representation_input_size, fc_representation_layers,
                                                            self.encoded_state_size,
                                                            init_zero=init_zero)
        elif 'concatted' in self.representation_type:
            self.representation_network = torch.nn.Identity()
            self.encoded_state_size = self.env_size * 2
            self.dynamics_input_size = self.encoded_state_size + action_space_size
            hidden_state_shape = None
        else:
            raise ValueError(f" self.representation_type = {self.representation_type} and should be one of the "
                             f"following: 'identity' 'encoder' 'concatted' 'learned'")

        self.policy_network = mlp(self.encoded_state_size, fc_policy_layers, action_space_size, init_zero=init_zero,
                                  momentum=momentum)

        if 'ensemble' in uncertainty_type:
            self.value_network = nn.ModuleList([mlp(self.encoded_state_size, fc_value_layers, value_support_size,
                                                    init_zero=init_zero, momentum=momentum)
                                                for _ in range(ensemble_size)])
            if self.use_prior:
                self.value_network_prior = nn.ModuleList([mlp(self.encoded_state_size, fc_value_prior_layers,
                                                              value_support_size, init_zero=False,
                                                              momentum=momentum)
                                                          for _ in range(ensemble_size)])
        else:
            self.value_network = mlp(self.encoded_state_size, fc_value_layers, value_support_size, init_zero=init_zero,
                                     momentum=momentum)

        self.dynamics_network = FullyConnectedDynamicsNetwork(
            self.env_size,
            mapping_seed,
            randomize_actions,
            self.dynamics_input_size,
            self.encoded_state_size,
            hidden_state_shape,
            fc_state_prediction_layers,
            fc_state_prediction_prior_layers,
            fc_reward_layers,
            fc_reward_prior_layers,
            reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=momentum,
            init_zero=init_zero,
            learned_model=self.learned_model,
            ensemble=True if 'ensemble' in uncertainty_type else False,
            ensemble_size=self.ensemble_size,
            use_prior=self.use_prior,
            prior_scale=self.prior_scale,
            use_encoder=self.use_encoder,
            representation_encoder=self.representation_encoder if not self.learned_model and self.use_encoder else None,
        )

        # projection
        num_channels = 1
        in_dim = num_channels * observation_shape[1] * observation_shape[2]
        self.porjection_in_dim = in_dim
        self.projection = torch.nn.Identity()
        self.projection_head = torch.nn.Identity()

        # RND
        if 'rnd' in uncertainty_type:
            self.rnd_scale = rnd_scale
            self.input_size_value_rnd = self.encoded_state_size
            self.input_size_reward_rnd = self.dynamics_input_size
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

        if 'ube' in uncertainty_type:
            self.inverse_ube_transform = inverse_ube_transform
            self.categorical_ube = categorical_ube
            if self.categorical_ube:
                assert inverse_ube_transform is not None and ube_support_size is not None, \
                    f"Can instantiate categorical UBE only if ube_support_size and inverse_ube_transform are " \
                    f"specified. inverse_ube_transform is None = {inverse_ube_transform is None}, " \
                    f"ube_support_size = {ube_support_size}"
                self.ube_network = no_batch_norm_mlp(self.encoded_state_size, fc_ube_layers, ube_support_size,
                                                     init_zero=False)
            else:
                # We don't initialize ube with zeros because we don't want to penalize the uncertainty of unobserved states
                self.ube_network = no_batch_norm_mlp(self.encoded_state_size, fc_ube_layers, 1,
                                                     init_zero=init_zero)  # , momentum=momentum)

    def representation(self, observation):
        # Regardless of the number of stacked observations, we only pass the last
        # observation = observation[:, -1, :, :].unsqueeze(1)
        # Should be faster and doing the same:
        observation = observation[:, -1:, :, :]

        # With the fully-connected deep_sea architecture, we maintain the original observation as the representation
        if self.use_identity_representation:
            encoded_state = self.representation_network(observation)
            # We only encode in representation if learned model is used. Otherwise, we pass the true state and encode it in
            # prediction and dynamics locally, to enable planning with the true model.
            if self.use_encoder and self.learned_model:
                # Flatten
                encoded_state = encoded_state.reshape(encoded_state.shape[0], -1)
                # Encode
                encoded_state = self.representation_encoder(encoded_state.detach()).detach()
                # Reshape back to [B , C * S, H, W] structure -> [B, 1, encoding_size, encoding_size]
                encoded_state = encoded_state.reshape(encoded_state.shape[0], 1, self.encoding_size, self.encoding_size)
            elif self.learned_model:
                # Rescale the states to make them easier to learn with the MSE-based consistency loss.
                encoded_state = encoded_state * self.amplify_one_hot
        elif 'concatted' in self.representation_type:
            device = observation.device
            N = self.env_size
            B = observation.shape[0]
            flattened_observation = observation.reshape(B, -1)
            mask = flattened_observation.sum(-1) == 0   # To identify zero-observations
            zeros_one_hot = torch.zeros(2 * N).to(device)

            # Get the indexes of the 1 hot along the B dimension, and shape them to 1 dim vectors of size batch_size
            rows = (flattened_observation.argmax(dim=-1) // N).long().squeeze(-1).to(device)
            columns = (flattened_observation.argmax(dim=-1) % N).long().squeeze(-1).to(device)
            one_hot_rows = torch.nn.functional.one_hot(rows, N).to(device)
            one_hot_cols = torch.nn.functional.one_hot(columns, N).to(device)

            # Setup the state tensor of shape [..., 2 * N]
            encoded_state = torch.cat((one_hot_rows[..., :N], one_hot_cols[..., :N]), dim=-1).float().to(device)
            # Setup all empty-state tensors to a zeros tensor of the right shape
            encoded_state[mask] = zeros_one_hot
            # Multiply by a constant to make it easy for MSE
            encoded_state = encoded_state * self.amplify_one_hot
        else:
            batch_size = observation.shape[0]
            observation = observation.reshape(observation.shape[0], -1)
            encoded_state = self.representation_network(observation).reshape(batch_size, 1, self.encoding_size,
                                                                             self.encoding_size)

        return encoded_state

    def prediction(self, encoded_state):
        # We reshape the encoded_state to the shape of input of the FC nets that follow
        encoded_state = encoded_state.reshape(encoded_state.shape[0], -1)

        # If we use an encoder, we plan with the true model, and this call came out of recurrent inference
        # i.e. encoded_state is a true state, we pass it through the encoder first.
        if self.use_encoder and encoded_state.shape[-1] != self.encoded_state_size and not self.learned_model:
            encoded_state = self.representation_encoder(encoded_state.detach()).detach()

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
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(reward_hidden, encoded_state, action)
        return next_encoded_state, reward_hidden, value_prefix

    def get_params_mean(self):
        if 'learned' in self.representation_type:
            representation_mean = []
            for name, param in self.representation_network.named_parameters():
                representation_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
            representation_mean = sum(representation_mean) / len(representation_mean)
        else:
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
        if self.learned_model and ('concatted' in self.representation_type or 'identity' in self.representation_type):
            state = state * (1 / self.amplify_one_hot)
        return self.rnd_scale * torch.nn.functional.mse_loss(self.value_rnd_network(state),
                                                             self.value_rnd_target_network(state).detach(),
                                                             reduction='none').sum(dim=-1)

    def compute_reward_rnd_uncertainty(self, state, action):
        flattened_state = state.reshape(-1, self.input_size_reward_rnd - self.action_space_size)
        if self.learned_model and ('concatted' in self.representation_type or 'identity' in self.representation_type):
            flattened_state = flattened_state * (1 / self.amplify_one_hot)

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
        action_one_hot.scatter_(1, action.long(), 1.0)
        if self.learned_model:
            action_one_hot = action_one_hot

        state_action = torch.cat((flattened_state, action_one_hot), dim=1).detach()

        return self.rnd_scale * torch.nn.functional.mse_loss(self.reward_rnd_network(state_action),
                                                             self.reward_rnd_target_network(state_action).detach(),
                                                             reduction='none').sum(dim=-1)

    def compute_ube_uncertainty(self, state):
        """
            Returns the value-uncertainty prediction from the UBE network (See UBE paper
            https://arxiv.org/pdf/1709.05380.pdf). The state is always detached, because we don't want the UBE
            prediction to train the learned dynamics and representation networks.
        """
        state = state.reshape(state.shape[0], -1)
        if self.use_encoder and not self.learned_model:
            state = self.representation_encoder(state.detach()).detach()
        # We squeeze the result to return tensor of shape [B] instead of [B, 1]
        ube_prediction = self.ube_network(state)
        # To guarantee that output value is positive, we treat it as a logit instead of as a direct scalar
        if not self.categorical_ube:
            ube_prediction = torch.exp(ube_prediction.squeeze(-1))
        return ube_prediction

    def rnd_ube_parameters(self):
        return itertools.chain(self.reward_rnd_network.parameters(), self.value_rnd_network.parameters(),
                               self.ube_network.parameters())

    def rnd_parameters(self):
        return itertools.chain(self.reward_rnd_network.parameters(), self.value_rnd_network.parameters())

    def other_parameters(self):
        return itertools.chain(self.representation_network.parameters(),
                               self.policy_network.parameters(),
                               self.value_network.parameters(),
                               self.dynamics_network.fc.parameters(),
                               # self.dynamics_network.state_prediction.parameters(),
                               self.projection.parameters(),
                               self.projection_head.parameters(),
                               # self.ube_network.parameters(),
                               # self.reward_rnd_network.parameters(), self.value_rnd_network.parameters()
                               )

    def ensemble_prediction_to_variance(self, logits):
        assert isinstance(logits, list), f"ensemble_prediction_to_variance was called on input that is not a list. " \
                                         f"type(logits) = {type(logits)}"
        # logits is a list of length ensemble_size, of tensors of shape: (num_parallel_envs, full_support_size)
        with torch.no_grad():
            batch_size = logits[0].shape[0]
            # Softmax the logits. Can also log-softmax the logits, which will result in more variance but less reliable.
            logits = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]

            # Stack into a tensor to compute the variance using torch
            stacked_tensor = torch.stack(logits, dim=0)
            # Shape of stacked_tensor: (ensemble_size, num_parallel_envs, full_support_size)

            # Compute the per-entry variance over the dimension 0 (ensemble size)
            scalar_variance = torch.var(stacked_tensor, unbiased=False, dim=0)
            # Resulting shape of scalar_variance: (num_parallel_envs, full_support_size)

            # Sum the per-entry variance scores
            scalar_variance = scalar_variance.sum(-1).squeeze()
            # Resulting shape of scalar_variance: (num_parallel_envs)

            assert len(scalar_variance.shape) == 1 and scalar_variance.shape[0] == batch_size, \
                f"expected scalar_variance.shape = (batch_size), and got scalar_variance.shape = " \
                f"{scalar_variance.shape}, while batch_size = {batch_size}"

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
                 fc_state_prediction_prior_layers,
                 fc_reward_layers,
                 fc_reward_prior_layers,
                 full_support_size,
                 lstm_hidden_size=64,
                 momentum=0.1,
                 init_zero=False,
                 learned_model=True,
                 ensemble=False,
                 ensemble_size=5,
                 use_prior=True,
                 prior_scale=10.0,
                 use_encoder=False,
                 representation_encoder=None,
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
        self.hidden_state_size = hidden_state_size
        self.dynamics_input_size = dynamics_input_size
        # True-model planning params
        self.env_size = env_size
        self.learned_model = learned_model
        # Ensemble params
        self.ensemble_size = ensemble_size
        self.ensemble = ensemble
        self.use_prior = use_prior
        self.prior_scale = prior_scale
        self.state_prediction_net_prior = None
        self.use_encoder = use_encoder
        self.representation_encoder = representation_encoder

        # Init dynamics prediction, learned or given
        if learned_model:
            self.state_prediction_net = no_batch_norm_mlp(dynamics_input_size, fc_state_prediction_layers,
                                                          hidden_state_size, init_zero=init_zero)
            # if self.use_prior:
            #     self.state_prediction_net_prior = no_batch_norm_mlp(dynamics_input_size, fc_state_prediction_prior_layers,
            #                                               hidden_state_size, init_zero=False)

        self.action_mapping = torch.from_numpy(DeepSea(size=env_size, mapping_seed=mapping_seed, seed=mapping_seed,
                                                       randomize_actions=randomize_actions)._action_mapping).long()
        self.action_mapping = self.action_mapping * 2 - 1

        # The architecture of the reward prediction is different between ensemble and not-ensemble.
        # To limit computation cost ensembles do not use EfficientZero's batch-norm / LSTM architecture.
        if self.ensemble:
            self.fc = nn.ModuleList([mlp(dynamics_input_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                                         momentum=momentum)
                                     for _ in range(ensemble_size)])
            if self.use_prior:
                self.fc_net_prior = nn.ModuleList([mlp(dynamics_input_size, fc_reward_prior_layers, full_support_size,
                                                       init_zero=False, momentum=momentum)
                                                   for _ in range(ensemble_size)])
            self.lstm = None
            self.bn_value_prefix = None
        else:
            # The input to the lstm is the concat tensor of hidden_state and action
            self.lstm = nn.LSTM(input_size=dynamics_input_size, hidden_size=lstm_hidden_size)
            self.bn_value_prefix = nn.BatchNorm1d(lstm_hidden_size, momentum=momentum)
            self.fc = mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                          momentum=momentum)
            # self.fc = no_batch_norm_mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero)

    def forward(self, reward_hidden, encoded_state, action):
        if self.learned_model:
            batch_size = encoded_state.shape[0]
            action_space_size = 2
            action_one_hot = (
                torch.zeros(
                    size=(
                        batch_size,  # batch dimension
                        action_space_size  # action space size
                    )
                )
                .to(action.device)
                .float()
            )
            action_one_hot.scatter_(1, action.long(), 1.0)
            flattened_state = encoded_state.reshape(batch_size, -1)
            x = torch.cat((flattened_state, action_one_hot), dim=-1)

            next_state = self.state_prediction_net(x) * 10

            if self.hidden_state_shape is not None:
                next_state = next_state.reshape(-1, self.hidden_state_shape[0], self.hidden_state_shape[1],
                                                self.hidden_state_shape[2])
        else:
            # Expected encoded_state shape: [num_envs or batch_size, channels, H, W] = [B, 1, env_size, env_size]
            next_state = self.get_batched_next_states(encoded_state, action)

        if self.use_encoder and not self.learned_model:
            # Flatten current state
            encoded_state = encoded_state.reshape(encoded_state.shape[0], -1)
            encoded_state = self.representation_encoder(encoded_state.detach()).detach()
            encoded_state_size = self.hidden_state_shape[0] * self.hidden_state_shape[1] * self.hidden_state_shape[2]
            action_one_hot = (
                torch.zeros(
                    (
                        encoded_state.shape[0],  # batch dimension
                        self.dynamics_input_size - encoded_state_size    # Action space size
                    )
                )
                .to(action.device)
                .float()
            )
            action_one_hot.scatter_(1, action.long(), 1.0)
            flattened_state = encoded_state.reshape(-1, encoded_state_size)
            x = torch.cat((flattened_state, action_one_hot), dim=1).detach()
        elif not self.learned_model:
            # Turn in a state_action vector
            batch_size = encoded_state.shape[0]
            action_one_hot = (
                torch.zeros(
                    size=(
                        batch_size,  # batch dimension
                        2   # action space size
                    )
                )
                .to(action.device)
                .float()
            )
            action_one_hot.scatter_(1, action.long(), 1.0)
            flattened_state = encoded_state.reshape(batch_size, -1)
            x = torch.cat((flattened_state, action_one_hot), dim=-1)

        # Reward prediction is done based on EfficientExplore architecture, only if we don't use an ensemble.
        if self.ensemble:
            if self.use_prior:
                value_prefix = [value_prefix_net(x) + self.prior_scale *
                                prior_net(x.detach()).detach() for value_prefix_net, prior_net
                                in zip(self.fc, self.fc_net_prior)]
            else:
                value_prefix = [value_prefix_net(x) for value_prefix_net in self.fc]
        else:
            x = x.unsqueeze(0)
            value_prefix, reward_hidden = self.lstm(x, reward_hidden)
            value_prefix = value_prefix.squeeze(0)
            value_prefix = self.bn_value_prefix(value_prefix)
            value_prefix = nn.functional.relu(value_prefix)
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

    def get_actions_right(self, batched_states):
        """
            Shape of batched_states is (B, H, W)
            Shape of batched_actions is (B, 1)
        """
        batch_size = batched_states.shape[0]
        N = batched_states.shape[-1]

        # Flatten 1 hot representation
        flattened_batched_states = batched_states.reshape(shape=(batch_size, N * N)).to(batched_states.device)

        # Get the indexes of the 1 hot along the B dimension, and shape them to 1 dim vectors of size batch_size
        rows, columns = (flattened_batched_states.argmax(dim=1) // N).long().squeeze(-1).to(batched_states.device), (
                flattened_batched_states.argmax(dim=1) % N).long().squeeze(-1).to(batched_states.device)

        # action_mapping expected to already be in the form -1, +1, transform back to 1-0
        action_mapping = (self.action_mapping + 1) / 2

        # Expand the action mapping
        action_mapping = action_mapping.unsqueeze(0).expand(batch_size, -1, -1).to(batched_states.device)

        # Flatten to shape [B, N * N]
        flattened_action_mapping = action_mapping.reshape(batch_size, N * N).to(batched_states.device)

        # Get all the actions_right with gather in the flattened (row, col) indexes.
        flattened_indexes = (rows * N + columns).to(batched_states.device)
        # actions_right should be of shape [B]
        actions_right = flattened_action_mapping.take(flattened_indexes).to(batched_states.device)

        return actions_right

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
        rows, columns = (flattened_batched_states.argmax(dim=1) // N).long().squeeze(-1).to(batched_states.device), (
                flattened_batched_states.argmax(dim=1) % N).long().squeeze(-1).to(batched_states.device)

        # Switch actions to -1, +1
        batched_actions = batched_actions.squeeze(-1) * 2 - 1

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


def normalize_along_last_dim(x):
    max_val = x.max(dim=-1, keepdim=True)[0]
    min_val = x.min(dim=-1, keepdim=True)[0]
    # Compute the range of values along the last dimension, avoiding division by zero
    range_val = torch.where((max_val - min_val) == 0, torch.tensor([1]).to(x.device), max_val - min_val)
    # Normalize the tensor along the last dimension, avoiding division by zero
    x_normalized = torch.where((max_val - min_val) == 0, x, (x - min_val) / range_val)

    return x_normalized
