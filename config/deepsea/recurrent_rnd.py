import torch
import torch.nn as nn
import itertools


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.ReLU,
        bias=True,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias), act()]
    return torch.nn.Sequential(*layers)


class FFResidualBlock(nn.Module):
    def __init__(self, flattened_input_length, output_size, hidden_layers):
        super().__init__()
        self.F = mlp(flattened_input_length, hidden_layers, output_size)

    def forward(self, x):
        identity = x
        out = self.F(x)
        out = nn.functional.relu(out)
        out = out + identity
        result = nn.functional.relu(out)

        return result

class SimpleRRND(nn.Module):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 output_size,
                 state_encoding_size,
                 target_network_layers,
                 encoder_layers,
                 dynamics_layers,
                 prediction_network_layers,
                 amplify_one_hot,
                 consistency_loss_coefficient,
                 dynamics_prior_coefficient,
                 representation_prior_coefficient,
                 train_encoder,
                 use_dynamics_prior=True,
                 use_representation_prior=True,
                 use_prediction_prior=True,
                 ):
        """
        """
        super().__init__()
        self.action_space_size = action_space_size
        self.flattened_observation_size = observation_shape[0] * observation_shape[1] * observation_shape[2]
        self.state_action_size = self.flattened_observation_size + action_space_size
        self.amplify_one_hot = amplify_one_hot
        self.consistency_loss_coefficient = consistency_loss_coefficient
        self.state_encoding_size = state_encoding_size
        self.dynamics_input_size = self.state_encoding_size + action_space_size
        self.dynamics_output_size = self.state_encoding_size
        self.dynamics_prior_coefficient = dynamics_prior_coefficient
        self.representation_prior_coefficient = representation_prior_coefficient
        self.train_encoder = train_encoder
        self.use_dynamics_prior = use_dynamics_prior
        self.use_representation_prior = use_representation_prior
        self.use_prediction_prior = use_prediction_prior

        self.target_network = mlp(self.state_action_size, target_network_layers, output_size)

        self.representation_1 = mlp(self.flattened_observation_size, encoder_layers, state_encoding_size)
        self.representation_2 = mlp(self.flattened_observation_size, encoder_layers, state_encoding_size)
        if self.use_representation_prior and self.train_encoder:
            self.representation_prior_1 = mlp(self.flattened_observation_size, encoder_layers, state_encoding_size)
            self.representation_prior_2 = mlp(self.flattened_observation_size, encoder_layers, state_encoding_size)

        self.dynamics_1 = mlp(self.dynamics_input_size, dynamics_layers, self.dynamics_output_size)
        self.dynamics_2 = mlp(self.dynamics_input_size, dynamics_layers, self.dynamics_output_size)

        if self.use_dynamics_prior:
            self.dynamics_prior_1 = mlp(self.dynamics_input_size, dynamics_layers, self.dynamics_output_size)
            self.dynamics_prior_2 = mlp(self.dynamics_input_size, dynamics_layers, self.dynamics_output_size)

        self.prediction_network_1 = mlp(self.dynamics_input_size, prediction_network_layers, output_size)
        self.prediction_network_2 = mlp(self.dynamics_input_size, prediction_network_layers, output_size)

        if self.use_prediction_prior:
            self.prediction_prior_1 = mlp(self.dynamics_input_size, prediction_network_layers, output_size)
            self.prediction_prior_2 = mlp(self.dynamics_input_size, prediction_network_layers, output_size)

    def compute_target(self, observation_batch, action):
        """
            Expecting observation_batch of shape [B, ...]
            And action batch of shape [B, 1]
        """
        with torch.no_grad():
            one_hot_action = self.action_to_one_hot(action)
            flattened_observation = observation_batch.reshape(observation_batch.shape[0], -1)
            state_action = torch.cat((flattened_observation, one_hot_action), dim=1)
            target_batch = self.target_network(state_action) * self.amplify_one_hot
            return target_batch

    def recurrent_rnd_uncertainty(self, action, hidden_states, rnd_scale=1.0):
        """
            Takes action batch and hidden states batch (h_1, h_2)
        """
        with torch.no_grad():
            prediction_1, hidden_1, prediction_2, hidden_2 = self.compute_rnd_predictions(action, hidden_states)
            uncertainty_prediction = rnd_scale * torch.nn.functional.mse_loss(prediction_1, prediction_2,
                                                                              reduction='none').sum(dim=-1)
            return uncertainty_prediction, (hidden_1, hidden_2)

    def compute_rnd_predictions(self, action, hidden_states):
        """
            hidden_states structure: (h_1, h_2)
        """
        one_hot_action = self.action_to_one_hot(action)
        assert len(one_hot_action.shape) == len(hidden_states[0].shape) == len(hidden_states[1].shape), \
            f"one_hot_action.shape = {one_hot_action.shape}, hidden_states[0].shape = {hidden_states[0].shape}, " \
            f"hidden_states[1].shape = {hidden_states[1].shape}"
        state_action_1 = torch.cat((hidden_states[0], one_hot_action), dim=-1)
        state_action_2 = torch.cat((hidden_states[1], one_hot_action), dim=-1)

        # Predict next hidden states
        if self.use_dynamics_prior:
            hidden_state_1 = self.dynamics_1(state_action_1) + \
                             self.dynamics_prior_coefficient * self.dynamics_prior_1(state_action_1.detach()).detach()
            hidden_state_2 = self.dynamics_2(state_action_2) + \
                             self.dynamics_prior_coefficient * self.dynamics_prior_2(state_action_2.detach()).detach()
        else:
            hidden_state_1 = self.dynamics_1(state_action_1)
            hidden_state_2 = self.dynamics_2(state_action_2)

        # Predict next rnd-predictions
        if self.use_prediction_prior:
            rnd_prediction_1 = (self.prediction_network_1(state_action_1) + self.representation_prior_coefficient *
                                self.prediction_prior_1(state_action_1.detach()).detach()) * self.amplify_one_hot
            rnd_prediction_2 = (self.prediction_network_2(state_action_2) + self.representation_prior_coefficient *
                                self.prediction_prior_2(state_action_2.detach()).detach()) * self.amplify_one_hot
        else:
            rnd_prediction_1 = self.prediction_network_1(state_action_1) * self.amplify_one_hot
            rnd_prediction_2 = self.prediction_network_2(state_action_2) * self.amplify_one_hot

        return rnd_prediction_1, hidden_state_1, rnd_prediction_2, hidden_state_2

    def get_total_loss(self, action, hidden_states, observation, is_initial_reprensentation, do_consistency=False):
        """
            The purpose of this function is to return a sum of all losses (even if initial, and also consistency if relevant)
            Parameters:
            action: The action chosen at hidden_states / observation
            hidden_states: the hidden rnd states. If None, this is an initial call at the start of an unroll
            observation: the observation from the environment that matches hidden_states
            next_observation: the observation from the environment after (observation, action)
            do_consistency: whether to train the rnd networks with consistency loss
        """
        target = self.compute_target(observation, action)
        loss, next_hidden_states = self.compute_rnd_prediction_loss(target, action, hidden_states)
        if do_consistency and not is_initial_reprensentation:
            loss += self.compute_consistency_loss(observation, hidden_states)
        return loss, next_hidden_states

    def compute_rnd_prediction_loss(self, target_prediction, action, hidden_states):
        """
            Takes two rnd-lstms, two hidden-states, one action and one target and computes the rnd loss, the two errors:
                loss = prediction_1 - target)**2 + (prediction_2 - target)**2
        """
        prediction_1, hidden_1, prediction_2, hidden_2 = self.compute_rnd_predictions(action, hidden_states)
        loss_one = torch.nn.functional.mse_loss(prediction_1, target_prediction.detach(),
                                                reduction='none').sum(dim=-1).squeeze()
        loss_two = torch.nn.functional.mse_loss(prediction_2, target_prediction.detach(),
                                                reduction='none').sum(dim=-1).squeeze()
        return loss_one + loss_two, (hidden_1, hidden_2)

    def representation(self, observation):
        batch_size = observation.shape[0]
        flattened_observation = observation.reshape(batch_size, -1) * self.amplify_one_hot

        if self.train_encoder:
            if self.use_representation_prior:
                hidden_state_1 = self.representation_1(flattened_observation) + \
                                 self.representation_prior_coefficient * \
                                 self.representation_prior_1(flattened_observation).detach()
                hidden_state_2 = self.representation_2(flattened_observation) + \
                                 self.representation_prior_coefficient * \
                                 self.representation_prior_2(flattened_observation).detach()
            else:
                hidden_state_1 = self.representation_1(flattened_observation)
                hidden_state_2 = self.representation_2(flattened_observation)
        else:
            with torch.no_grad():
                hidden_state_1 = self.representation_1(flattened_observation)
                # + self.representation_prior_coefficient * self.representation_prior_1(flattened_observation).detach()
                hidden_state_2 = self.representation_2(flattened_observation)
                # + self.representation_prior_coefficient * self.representation_prior_2(flattened_observation).detach()

        return hidden_state_1, hidden_state_2

    def compute_consistency_loss(self, current_observation, hidden_states):
        # Compute the initial hidden cell state c encoding for this state using the encoder, without gradient
        predicted_hidden_state_1, predicted_hidden_state_2 = hidden_states

        with torch.no_grad():
            target_hidden_state_1, target_hidden_state_2 = self.representation(current_observation)

        # Compute the loss between the hidden cell c and the predicted hidden cell c
        loss_1 = torch.nn.functional.mse_loss(predicted_hidden_state_1, target_hidden_state_1.detach(),
                                              reduction='none').sum(
            dim=-1).squeeze()
        loss_2 = torch.nn.functional.mse_loss(predicted_hidden_state_2, target_hidden_state_2.detach(),
                                              reduction='none').sum(
            dim=-1).squeeze()

        return (loss_1 + loss_2) * self.consistency_loss_coefficient

    def action_to_one_hot(self, action):
        """
            Takes an action batch of shape: [B, 1]
            And return a one-hot representation of shape: [B, action_space_size]
        """
        action_one_hot = (
            torch.zeros(
                (
                    action.shape[0],  # batch dimension
                    self.action_space_size
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)

        return action_one_hot

    def parameters(self):
        return itertools.chain(self.representation_1.parameters(),
                               self.representation_2.parameters(),
                               self.dynamics_1.parameters(),
                               self.dynamics_2.parameters(),
                               self.prediction_network_1.parameters(),
                               self.prediction_network_2.parameters(),
                               )

    def all_parameters(self):
        return itertools.chain(self.target_network.parameters(),
                               self.representation_1.parameters(),
                               self.representation_prior_1.parameters(),
                               self.representation_2.parameters(),
                               self.representation_prior_2.parameters(),
                               self.dynamics_1.parameters(),
                               self.dynamics_prior_1.parameters(),
                               self.dynamics_2.parameters(),
                               self.dynamics_prior_2.parameters(),
                               self.prediction_network_1.parameters(),
                               self.prediction_network_2.parameters(),
                               )


class ResNetRRND(SimpleRRND):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 output_size,
                 state_encoding_size,
                 target_network_layers,
                 encoder_layers,
                 dynamics_layers,
                 prediction_network_layers,
                 amplify_one_hot,
                 consistency_loss_coefficient,
                 dynamics_prior_coefficient,
                 representation_prior_coefficient,
                 train_encoder,
                 use_dynamics_prior=True,
                 use_representation_prior=True,
                 use_prediction_prior=True,
                 ):
        super().__init__(
            observation_shape,
            action_space_size,
            output_size,
            state_encoding_size,
            target_network_layers,
            encoder_layers,
            dynamics_layers,
            prediction_network_layers,
            amplify_one_hot,
            consistency_loss_coefficient,
            dynamics_prior_coefficient,
            representation_prior_coefficient,
            train_encoder,
            use_dynamics_prior=use_dynamics_prior,
            use_representation_prior=use_representation_prior,
            use_prediction_prior=use_prediction_prior,
        )

        self.dynamics_1 = nn.ModuleList(
            [
                FFResidualBlock(self.dynamics_input_size, self.dynamics_input_size, dynamics_layers[:-1]),
                mlp(self.dynamics_input_size, [dynamics_layers[-1]], self.dynamics_output_size)
            ]
        )
        self.dynamics_2 = nn.ModuleList(
            [
                FFResidualBlock(self.dynamics_input_size, self.dynamics_input_size, dynamics_layers[:-1]),
                mlp(self.dynamics_input_size, [dynamics_layers[-1]], self.dynamics_output_size)
            ]
        )

        if self.use_dynamics_prior:
            self.dynamics_prior_1 = nn.ModuleList(
                [
                    FFResidualBlock(self.dynamics_input_size, self.dynamics_input_size, dynamics_layers[:-1]),
                    mlp(self.dynamics_input_size, [dynamics_layers[-1]], self.dynamics_output_size)
                ]
            )
            self.dynamics_prior_2 = nn.ModuleList(
                [
                    FFResidualBlock(self.dynamics_input_size, self.dynamics_input_size, dynamics_layers[:-1]),
                    mlp(self.dynamics_input_size, [dynamics_layers[-1]], self.dynamics_output_size)
                ]
            )

    def compute_rnd_predictions(self, action, hidden_states):
        """
            hidden_states structure: (h_1, h_2)
        """
        one_hot_action = self.action_to_one_hot(action)
        assert len(one_hot_action.shape) == len(hidden_states[0].shape) == len(hidden_states[1].shape), \
            f"one_hot_action.shape = {one_hot_action.shape}, hidden_states[0].shape = {hidden_states[0].shape}, " \
            f"hidden_states[1].shape = {hidden_states[1].shape}"
        state_action_1 = torch.cat((hidden_states[0], one_hot_action), dim=-1)
        state_action_2 = torch.cat((hidden_states[1], one_hot_action), dim=-1)

        # Predict next hidden states
        hidden_state_1 = state_action_1
        for block in self.dynamics_1:
            hidden_state_1 = block(hidden_state_1)

        hidden_state_2 = state_action_2
        for block in self.dynamics_2:
            hidden_state_2 = block(hidden_state_2)

        if self.use_dynamics_prior:
            with torch.no_grad():
                prior_hidden_state_1 = state_action_1
                for block in self.dynamics_prior_1:
                    prior_hidden_state_1 = block(prior_hidden_state_1)

                prior_hidden_state_2 = state_action_2
                for block in self.dynamics_prior_2:
                    prior_hidden_state_2 = block(prior_hidden_state_2)

            hidden_state_1 = hidden_state_1 + self.dynamics_prior_coefficient * prior_hidden_state_1.detach()
            hidden_state_2 = hidden_state_2 + self.dynamics_prior_coefficient * prior_hidden_state_2.detach()

        # Predict next rnd-predictions
        if self.use_prediction_prior:
            rnd_prediction_1 = (self.prediction_network_1(state_action_1) + self.representation_prior_coefficient *
                                self.prediction_prior_1(state_action_1.detach()).detach()) * self.amplify_one_hot
            rnd_prediction_2 = (self.prediction_network_2(state_action_2) + self.representation_prior_coefficient *
                                self.prediction_prior_2(state_action_2.detach()).detach()) * self.amplify_one_hot
        else:
            rnd_prediction_1 = self.prediction_network_1(state_action_1) * self.amplify_one_hot
            rnd_prediction_2 = self.prediction_network_2(state_action_2) * self.amplify_one_hot

        return rnd_prediction_1, hidden_state_1, rnd_prediction_2, hidden_state_2



