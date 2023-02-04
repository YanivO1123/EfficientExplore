import torch
import typing

import numpy as np
import torch.nn as nn

from typing import List


class NetworkOutput(typing.NamedTuple):
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden: object
    value_variance: float
    value_prefix_variance: float


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst):
    #TODO: This is used by reanalyze to compute fresh predictions. In reanlyze, in basic MuExplore,
    # the variances are not used (they are used to provide targets for UBE, if UBE is used)

    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst =[], []
    # value_variance_lst, value_prefix_variance_list = [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.value_prefix)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))
        # value_variance_lst.append(output.value_variance)
        # value_prefix_variance_list.append(output.value_prefix_variance)

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    # hidden_state_lst = torch.cat(hidden_state_lst, 0)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)
    # value_variance_lst = np.concatenate(value_variance_lst)
    # value_prefix_variance_list = np.concatenate(value_prefix_variance_list)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (reward_hidden_c_lst, reward_hidden_h_lst)


class BaseNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform, lstm_hidden_size):
        """Base Network
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        """
        super(BaseNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.lstm_hidden_size = lstm_hidden_size
        self.uncertainty_type = None

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def ensemble_prediction_to_variance(self, logits):
        raise NotImplementedError

    def compute_rnd_uncertainty(self, state):
        raise NotImplementedError

    def compute_ube_uncertainty(self, state):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)
        
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)
        value_variance = None
        value_prefix_variance = None
        if self.uncertainty_type == 'rnd_ube' or self.uncertainty_type == 'ensemble_ube':
            value_variance = self.compute_ube_uncertainty(state.detach())

        if not self.training:
            #MuExplore: Compute the variance of the value prediction, and set the variance of the value_prefix
            if self.uncertainty_type == 'ensemble':  # If the ensemble arch. is used
                value_variance = self.ensemble_prediction_to_variance(value).detach().cpu().numpy()
            elif self.uncertainty_type == 'rnd':
                value_variance = self.compute_rnd_uncertainty(state.detach()).detach().cpu().numpy()
            elif self.uncertainty_type == 'rnd_ube' or self.uncertainty_type == 'ensemble_ube':
                value_variance = self.compute_ube_uncertainty(state.detach()).detach().cpu().numpy()
            value_prefix_variance = [0. for _ in range(num)]

            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            actor_logit = actor_logit.detach().cpu().numpy()
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                             torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy())
        else:
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'), torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))

        return NetworkOutput(value, [0. for _ in range(num)], actor_logit, state, reward_hidden, value_variance, value_prefix_variance)

    def recurrent_inference(self, hidden_state, reward_hidden, action) -> NetworkOutput:
        state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        actor_logit, value = self.prediction(state)

        #MuExplore: Setup the uncertainty return values, which are only relevant when not-training
        value_variance = None
        value_prefix_variance = None

        if self.uncertainty_type == 'rnd_ube' or self.uncertainty_type == 'ensemble_ube':
            value_variance = self.compute_ube_uncertainty(state.detach())

        if not self.training:
            # MuExplore: Compute the variance of the value prediction, and set the variance of the value_prefix
            if self.uncertainty_type == 'ensemble':  # If the ensemble arch. is used
                if isinstance(value, list):  # If the ensemble arch. is used
                    value_variance = self.ensemble_prediction_to_variance(value).detach().cpu().numpy()
                if isinstance(value_prefix, list):  # If the ensemble arch. is used
                    value_prefix_variance = self.ensemble_prediction_to_variance(value_prefix).detach().cpu().numpy()
            elif self.uncertainty_type == 'rnd':
                value_variance = self.compute_rnd_uncertainty(state.detach()).detach().cpu().numpy()
                value_prefix_variance = self.compute_rnd_uncertainty(state.detach())
            elif self.uncertainty_type == 'rnd_ube' or self.uncertainty_type == 'ensemble_ube':
                value_variance = self.compute_ube_uncertainty(state.detach()).detach().cpu().numpy()
                value_prefix_variance = self.compute_rnd_uncertainty(state.detach()).detach().cpu().numpy()

            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            value_prefix = self.inverse_reward_transform(value_prefix).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            reward_hidden = (reward_hidden[0].detach().cpu().numpy(), reward_hidden[1].detach().cpu().numpy())
            actor_logit = actor_logit.detach().cpu().numpy()

        return NetworkOutput(value, value_prefix, actor_logit, state, reward_hidden, value_variance, value_prefix_variance)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

def renormalize(tensor, first_dim=1):
    # normalize the tensor (states)
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)

