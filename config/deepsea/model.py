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


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2, downsample=self.conv2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        downsample,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-1,:,:]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = nn.functional.relu(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)

        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        downsample,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_reward: int
            channels of reward head
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_blocks,
            num_channels,
            downsample,
            momentum=bn_mt,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels + 1,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        in_dim = num_channels * observation_shape[1] * observation_shape[2] # num_channels * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
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

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()


class EfficientExploreNet(EfficientZeroNet):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 num_blocks,
                 num_channels,
                 reduced_channels_reward,
                 reduced_channels_value,
                 reduced_channels_policy,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 fc_rnd_layers,
                 reward_support_size,
                 value_support_size,
                 downsample,
                 inverse_value_transform,
                 inverse_reward_transform,
                 lstm_hidden_size,
                 bn_mt=0.1,
                 proj_hid=256,
                 proj_out=256,
                 pred_hid=64,
                 pred_out=256,
                 init_zero=False,
                 state_norm=False,
                 ensemble_size=2,
                 use_network_prior=True,
                 prior_scale=10,
                 uncertainty_type='ensemble',
                 rnd_scale=1,
                 ):
        super(EfficientExploreNet, self).__init__(
            observation_shape,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            reduced_channels_value,
            reduced_channels_policy,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            reward_support_size,
            value_support_size,
            downsample,
            inverse_value_transform,
            inverse_reward_transform,
            lstm_hidden_size,
            bn_mt=bn_mt,
            proj_hid=proj_hid,
            proj_out=proj_out,
            pred_hid=pred_hid,
            pred_out=pred_out,
            init_zero=init_zero,
            state_norm=state_norm)

        self.uncertainty_type = uncertainty_type

        block_output_size_reward = (
            (
                    reduced_channels_reward
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                    reduced_channels_value
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                    reduced_channels_policy
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        if self.uncertainty_type == 'ensemble' or self.uncertainty_type == 'ensemble_ube':
            self.dynamics_network = EnsembleDynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                reward_support_size,
                block_output_size_reward,
                lstm_hidden_size=lstm_hidden_size,
                momentum=bn_mt,
                init_zero=self.init_zero,
                ensemble_size=ensemble_size,
                use_network_prior=use_network_prior,
                prior_scale=prior_scale,
            )

            self.prediction_network = EnsemblePredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                value_support_size,
                block_output_size_value,
                block_output_size_policy,
                momentum=bn_mt,
                init_zero=self.init_zero,
                ensemble_size=ensemble_size,
                use_network_prior=use_network_prior,
                prior_scale=prior_scale,
            )
        elif self.uncertainty_type == 'rnd' or self.uncertainty_type == 'rnd_ube':
            self.input_size_rnd = (
                (
                        num_channels
                        * math.ceil(observation_shape[1] / 16)
                        * math.ceil(observation_shape[2] / 16)
                )
                if downsample
                else (num_channels * observation_shape[1] * observation_shape[2])
            )
            self.rnd_scale = rnd_scale
            # It's important that the RND nets are NOT initiated with zero
            self.rnd_network = mlp(self.input_size_rnd, fc_rnd_layers[:-1], fc_rnd_layers[-1], init_zero=False, momentum=bn_mt)
            self.rnd_target_network = mlp(self.input_size_rnd, fc_rnd_layers[:-1], fc_rnd_layers[-1], init_zero=False, momentum=bn_mt)

        if self.uncertainty_type == 'ensemble_ube' or self.uncertainty_type == 'rnd_ube':
            self.use_ube = True
            self.ube_network = None
            raise NotImplementedError


    #TODO: Complete this function, and move it because it probably doesnt belong here.
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

    # TODO: Complete the UBE prediction computation
    def ube(self, encoded_state, action):
        # I need to decide if UBE takes encoded_state or encoded_state and action
        return NotImplementedError

    def compute_rnd_uncertainty(self, state):
        state = state.view(-1, self.input_size_rnd)
        return self.rnd_scale * torch.nn.functional.mse_loss(self.rnd_network(state),
                                                             self.rnd_target_network(state),
                                                             reduction='none')

class EnsembleDynamicsNetwork(DynamicsNetwork):
    def __init__(
            self,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
            lstm_hidden_size=64,
            momentum=0.1,
            init_zero=False,
            ensemble_size=2,
            use_network_prior=True,
            prior_scale = 10,
    ):
        """
            The EnsembleDynamicsNetwork shares the same architecture with the DynamicsNetwork, with the exception that
            the EnsembleDynamicsNetwork has an ensemble of value-prefix and reward-hidden prediction heads.
            These heads are a list of LSTMs, a list of bn_value_prefix networks, and fc networks
        """
        super(EnsembleDynamicsNetwork, self).__init__(
            num_blocks,
            num_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
            lstm_hidden_size,
            momentum,
            init_zero)

        self.ensemble_size = ensemble_size
        self.use_network_prior = use_network_prior
        self.prior_scale = prior_scale

        # self.bn_value_prefix_nets = nn.ModuleList([nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum) for _ in range(ensemble_size)])
        # self.bn_value_prefix = None # To make sure there are no unused params
        self.fc_nets = nn.ModuleList([mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum) for _ in range(ensemble_size)])
        self.fc = None  # To make sure there are no unused params

        if self.use_network_prior:
            self.prior_fc_nets = nn.ModuleList([mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum) for _ in range(ensemble_size)])
        else:
            self.prior_fc_nets = None

    def forward(self, x, reward_hidden):
        """
            The ensemble dynamics network returns a list output value_prefix_list
        """
        state = x[:, :-1, :, :]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)

        # When bn_value_prefix_nets are part of the ensemble:
        # original_value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        # original_value_prefix = original_value_prefix.squeeze(0)
        # value_prefix_list = [net(original_value_prefix) for net in self.bn_value_prefix_nets]
        # value_prefix_list = [nn.functional.relu(value_prefix) for value_prefix in value_prefix_list]

        #Otherwise:
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        if self.use_network_prior:
            value_prefix_list = [fc_net(value_prefix) + self.prior_scale * prior_fc_net(value_prefix.detach()).detach() for fc_net, prior_fc_net in zip(self.fc_nets, self.prior_fc_nets)]
        else:
            value_prefix_list = [fc_net(value_prefix) for fc_net in self.fc_nets]

        return state, reward_hidden, value_prefix_list

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)
        return_reward_w_dist = None
        for index, fc_net in enumerate(self.fc_nets):
            for name, param in fc_net.named_parameters():
                temp_weights = param.detach().cpu().numpy().reshape(-1)
                reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
            if index == 0:
                return_reward_w_dist = reward_w_dist

        reward_mean = np.abs(reward_w_dist).mean()

        return return_reward_w_dist, reward_mean

class EnsemblePredictionNetwork(PredictionNetwork):
    def __init__(
            self,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=0.1,
            init_zero=False,
            ensemble_size=2,
            use_network_prior=True,
            prior_scale=10,
    ):
        """Ensembled Prediction network
        Parameters (additional to Prediction network)
        ----------
        ensemble_size: int
            the size of the ensemble
        use_network_prior: bool
            whether to use a prior-network, see Randomized Prior Functions for Deep Reinforcement Learning
        prior_scale: int
            The scale of the influence of the prior network on the prediction
        """
        super(EnsemblePredictionNetwork, self).__init__(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=momentum,
            init_zero=init_zero,
        )

        self.ensemble_size = ensemble_size
        self.prior_scale = prior_scale
        self.use_network_prior = use_network_prior
        # self.bn_value_nets = nn.ModuleList([nn.BatchNorm2d(reduced_channels_value, momentum=momentum) for _ in range(ensemble_size)])
        # self.bn_value = None
        self.fc_value_nets = nn.ModuleList([mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum) for _ in range(ensemble_size)])
        self.fc_value = None

        if self.use_network_prior:
            self.prior_fc_value_nets = nn.ModuleList([mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum) for _ in range(ensemble_size)])
        else:
            self.prior_fc_value_nets = None

    def forward(self, x):
        """
            This function takes as input a state x, and outputs a LIST of tensors values, and a tesnor policy
        """
        for block in self.resblocks:
            x = block(x)
        value_after_conv = self.conv1x1_value(x)
        # If the bn layers are not ensembled:
        if self.bn_value is not None:
            value = self.bn_value(value_after_conv)
            value = nn.functional.relu(value)
        else:
            values = [bn_value_net(value_after_conv) for bn_value_net in self.bn_value_nets]
            values = [nn.functional.relu(value) for value in values]

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)

        # If the bn layers are not ensembled:
        if self.bn_value is not None:
            value = value.view(-1, self.block_output_size_value)
        else:
            values = [value.view(-1, self.block_output_size_value) for value in values]
        policy = policy.view(-1, self.block_output_size_policy)

        # If the bn layers are not ensembled:
        if self.bn_value is not None:
            if self.use_network_prior:
                # This can also be done with torch.no_grad()
                values = [fc_value_net(value) + self.prior_scale * prior_fc_value_net(value.detach()).detach() for fc_value_net, prior_fc_value_net in zip(self.fc_value_nets, self.prior_fc_value_nets)]
            else:
                values = [fc_value_net(value) for fc_value_net in self.fc_value_nets]
        else:
            if self.use_network_prior:
                # This can also be done with torch.no_grad()
                values = [fc_value_net(value) + self.prior_scale * prior_fc_value_net(value.detach()).detach() for fc_value_net, prior_fc_value_net, value in zip(self.fc_value_nets, self.prior_fc_value_nets, values)]
            else:
                values = [fc_value_net(value) for fc_value_net, value in zip(self.fc_value_nets, values)]

        policy = self.fc_policy(policy)

        return policy, values

class FullyConnectedEfficientZeroNet(BaseNet):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 fc_state_prediction_layers,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 fc_rnd_layers,
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
                 ):
        """
            FullyConnected (or, non-resnet) EfficientZero network.
            Parameters
            __________
            observation_shape: tuple or list
                shape of observations: [C, W, H] = [1, N, N], for deep sea for which this arch. is implemented.

        """
        super(FullyConnectedEfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.action_space_size = action_space_size
        # The size of flattened encoded state:
        self.encoded_state_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
        # The size of the input to the dynamics network is (num_channels + the action channel) * H * W
        self.dynamics_input_size = (observation_shape[0] + 1) * observation_shape[1] * observation_shape[2]
        # In this arch. the representation is the original observation
        self.representation_network = torch.nn.Identity()
        self.policy_network = mlp(self.encoded_state_size, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)
        self.value_network = mlp(self.encoded_state_size, fc_value_layers, value_support_size, init_zero=init_zero, momentum=momentum)
        self.learned_model = learned_model
        if learned_model:
            self.dynamics_network = FullyConnectedDynamicsNetwork(
                self.dynamics_input_size,
                self.encoded_state_size,
                observation_shape,
                fc_state_prediction_layers,
                fc_reward_layers,
                reward_support_size,
                lstm_hidden_size=lstm_hidden_size,
                momentum=momentum,
                init_zero=init_zero,
            )
        else:
            self.dynamics_network = DeepSeaDynamicsNetwork(
                env_size,
                mapping_seed,
                randomize_actions,
                self.dynamics_input_size,
                observation_shape,
                fc_reward_layers,
                reward_support_size,
                lstm_hidden_size=lstm_hidden_size,
                momentum=momentum,
                init_zero=init_zero,
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
        self.input_size_rnd = self.encoded_state_size
        self.rnd_scale = rnd_scale
        # It's important that the RND nets are NOT initiated with zero
        self.rnd_network = mlp(self.input_size_rnd, fc_rnd_layers[:-1], fc_rnd_layers[-1], init_zero=False,
                               momentum=momentum)
        self.rnd_target_network = mlp(self.input_size_rnd, fc_rnd_layers[:-1], fc_rnd_layers[-1], init_zero=False,
                                      momentum=momentum)

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
        value = self.value_network(encoded_state)
        return policy, value

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],     # batch dimension
                    1,                          # channels dimension
                    encoded_state.shape[2],     # H dim
                    encoded_state.shape[3],     # W dim
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
            next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, encoded_state, action,
                                                                                    reward_hidden)
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

    def compute_rnd_uncertainty(self, state):
        state = state.view(-1, self.input_size_rnd)
        return self.rnd_scale * torch.nn.functional.mse_loss(self.rnd_network(state),
                                                             self.rnd_target_network(state),
                                                             reduction='none')

class FullyConnectedDynamicsNetwork(nn.Module):
    def __init__(self,
                 dynamics_input_size,
                 hidden_state_size,
                 hidden_state_shape,
                 fc_state_prediction_layers,
                 fc_reward_layers,
                 full_support_size,
                 lstm_hidden_size=64,
                 momentum=0.1,
                 init_zero=False,
                 ):
        """
        Non-resnet, non-conv dynamics network
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
        self.state_prediction_net = mlp(dynamics_input_size, fc_state_prediction_layers, hidden_state_size, init_zero=init_zero,
                                        momentum=momentum)
        # The input to the lstm is the concat tensor of hidden_state and action
        self.lstm = nn.LSTM(input_size=dynamics_input_size, hidden_size=lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(lstm_hidden_size, momentum=momentum)
        self.fc = mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                      momentum=momentum)

    def forward(self, x, reward_hidden):
        # Flatten input state-action for FC nets
        x = x.view(-1, self.dynamics_input_size)

        # Next-state prediction is done based on a FC network
        next_state = self.state_prediction_net(x)
        next_state = nn.functional.relu(next_state)
        # Reshape the state to the shape MuZero expects: [num_envs or batch_size, channels, H, W] = [B, 1, N, N]
        next_state = next_state.view(-1,
                                     self.hidden_state_shape[0],
                                     self.hidden_state_shape[1],
                                     self.hidden_state_shape[2])

        # Reward prediction is done based on EfficientExplore architecture
        x = x.unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return next_state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = []
        for name, param in self.state_prediction_net.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
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

class DeepSeaDynamicsNetwork(nn.Module):
    def __init__(self,
                 env_size,
                 mapping_seed,
                 randomize_actions,
                 dynamics_input_size,
                 hidden_state_shape,
                 fc_reward_layers,
                 full_support_size,
                 lstm_hidden_size=64,
                 momentum=0.1,
                 init_zero=False,
                 ):
        super().__init__()
        self.env_size = env_size
        self.hidden_state_shape = hidden_state_shape
        self.dynamics_input_size = dynamics_input_size
        # The input to the lstm is the concat tensor of hidden_state and action
        self.lstm = nn.LSTM(input_size=dynamics_input_size, hidden_size=lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(lstm_hidden_size, momentum=momentum)
        self.fc = mlp(lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero,
                      momentum=momentum)
        # self.true_model = DeepSea(size=env_size, mapping_seed=mapping_seed, seed=mapping_seed,
        #                           randomize_actions=randomize_actions)
        self.action_mapping = torch.from_numpy(DeepSea(size=env_size, mapping_seed=mapping_seed, seed=mapping_seed,
                                  randomize_actions=randomize_actions)._action_mapping).long()
        self.action_mapping = self.action_mapping * 2 - 1

    def forward(self, x, current_state, action, reward_hidden):
        # Produce next states from previous state and action, in the shape MuZero expects:
        # [num_envs or batch_size, channels, H, W] = [B, 1, N, N]
        with torch.no_grad():
            next_state = self.get_batched_next_states(current_state, action)

        # Flatten input state-action for FC reward prediction nets
        x = x.view(-1, self.dynamics_input_size)

        # Reward prediction is done based on EfficientExplore architecture
        x = x.unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return next_state, reward_hidden, value_prefix

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

    def get_dynamic_mean(self):
        return 0

    def get_reward_mean(self):
        for index, (name, param) in enumerate(self.fc.named_parameters()):
            if index > 0:
                temp_weights = param.detach().cpu().numpy().reshape(-1)
                reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
            else:
                reward_w_dist = param.detach().cpu().numpy().reshape(-1)
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean
