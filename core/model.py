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
    rnd_hidden_state: List[float]


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output_value_variance(output_lst):
    # concat the value uncertainties of the model output list
    value_uncertainty_lst = []
    for output in output_lst:
        value_uncertainty_lst.append(output.value_variance)

    value_uncertainty_lst = np.concatenate(value_uncertainty_lst)

    return value_uncertainty_lst


def concat_output_reward_variance(output_lst):
    # concat the value uncertainties of the model output list
    value_prefix_variance_lst = []
    for output in output_lst:
        value_prefix_variance_lst.append(output.value_prefix_variance)

    value_prefix_variance_lst = np.concatenate(value_prefix_variance_lst)

    return value_prefix_variance_lst


def concat_output(output_lst):
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.value_prefix)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (reward_hidden_c_lst, reward_hidden_h_lst)


def concat_uncertainty_output(output_lst):
    # concat the model output
    value_variance_lst, reward_variance_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst = [], []
    rnd_hidden_states = []

    for output in output_lst:
        value_variance_lst.append(output.value_variance)
        reward_variance_lst.append(output.value_prefix_variance)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))
        if output.rnd_hidden_state is not None:
            rnd_hidden_states.append(output.rnd_hidden_state)

    value_variance_lst = np.concatenate(value_variance_lst)
    reward_variance_lst = np.concatenate(reward_variance_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)
    if len(rnd_hidden_states) > 0:
        rnd_hidden_states = np.concatenate(rnd_hidden_states)

    return value_variance_lst, reward_variance_lst, policy_logits_lst, hidden_state_lst, (
        reward_hidden_c_lst, reward_hidden_h_lst), rnd_hidden_states


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
        # To be specified as an input to uncertainty nets
        self.uncertainty_type = None
        # To be computed in uncertainty nets that use RND
        self.value_uncertainty_propagation_scale = None
        # To be computed in uncertainty nets that use encoder
        self.learned_model = None
        self.use_encoder = None
        self.representation_encoder = None
        self.categorical_ube = True
        self.inverse_ube_transform = None

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def ensemble_prediction_to_variance(self, logits):
        raise NotImplementedError

    def compute_value_rnd_uncertainty(self, state):
        raise NotImplementedError

    def compute_reward_rnd_uncertainty(self, state, action):
        raise NotImplementedError

    def compute_ube_uncertainty(self, state):
        raise NotImplementedError

    def rnd_representation(self, obs_history):
        raise NotImplementedError

    def compute_uncertainty(self, previous_state=None, next_state=None, action=None, value=None, value_prefix=None,
                            previous_rnd_state=None, next_rnd_state=None):
        """
            This function deals with all the different cases of uncertainty estimation implemented.
            The following cases are implemented:
                RND for reward and value prediction (separately).
                RND for reward prediction and UBE based value-uncertainty prediction.
                Ensemble for reward and value prediction.
                Ensemble for reward prediction and UBE based value-uncertainty prediction.
                No uncertainty mechanism. This is used when MuExplore is not used, or when visitation counting is used
                as the sole uncertainty mechanism.
            When UBE is used, there is an additional value-uncertainty mechanism in both cases: either RND or ensemble,
            to increase the agent's sensitivity to new states.
            The value uncertainty with RND and UBE are predicted for the variable next_state.
            The reward uncertainty of RND is predicted for variables previous_state, action.
            Parameters:
            -----------
            previous_state: the state for which recurrent inf. predicts next state. Used in reward_rnd computation.
                of shape (B, C or C * stacked_obs, H, W)
            next state: the predicted next state from previous_state and action. Used in value_rnd computation.
                of shape (B, C or C * stacked_obs, H, W)
            action: action tensor of shape (B, 1), the action taken at (hidden) state previous_state.
                Used in reward_rnd computation.
            value: for ensemble-value computation, will be a list over value tensors of shape (B, support).
                Otherwise ignored.
            value_prefix: for ensemble-value_prefix computation, will be a list over value_prefix tensors of shape
                (B, support). Otherwise ignored.
            previous_rnd_state:
                When RND is trained with an independent dynamics network, the current hidden state of that network
                (matches action).
            next_rnd_state:
                When RND is trained with an independent dynamics network, the next hidden state of that network
                (result of state-action rnd-dynamics prediction).
        """
        with torch.no_grad():
            # If no uncertainty mech. is used but we're in initial inference we can return zeros for val. var. unc.
            if self.uncertainty_type is None or 'none' in self.uncertainty_type and previous_state is None:
                return None, np.zeros(next_state.shape[0])
            # If no uncertainty mech. is used, we will return Nones
            elif self.uncertainty_type is None or 'none' in self.uncertainty_type:
                return None, None
            else:
                batch_size = next_state.shape[0]
                value_variance = None
                value_prefix_variance = None

                # If the call came from initial_inference:
                if action is None and previous_state is None and next_state is not None:
                    # Reward-variance is zeros
                    value_prefix_variance = torch.tensor([0. for _ in range(batch_size)]).to(next_state.device)

                # Case ensemble
                if 'ensemble' in self.uncertainty_type:
                    if isinstance(value, list):
                        value_variance = self.ensemble_prediction_to_variance(value)
                    if isinstance(value_prefix, list):
                        value_prefix_variance = self.ensemble_prediction_to_variance(value_prefix).detach()
                elif 'double_model_rnd' in self.uncertainty_type:
                    if next_rnd_state is not None:
                        assert self.value_uncertainty_propagation_scale is not None
                        value_variance = self.compute_value_rnd_uncertainty(next_rnd_state)
                        if action is not None:
                            assert previous_rnd_state is not None
                            # Compute reward-unc. as epistemic unc. associated with NEXT state as well as transition
                            # (previous state + action)
                            value_prefix_variance = self.compute_reward_rnd_uncertainty(previous_rnd_state, action)
                # Case RND
                elif 'rnd' in self.uncertainty_type:
                    assert self.value_uncertainty_propagation_scale is not None
                    value_variance = self.compute_value_rnd_uncertainty(next_state)
                    if action is not None:
                        # Compute reward-unc. as epistemic unc. associated with NEXT state as well as transition
                        # (previous state + action)
                        value_prefix_variance = self.compute_reward_rnd_uncertainty(previous_state, action)
                # Cap local uncertainty at 1
                # value_prefix_variance = torch.minimum(value_prefix_variance, torch.ones_like(value_prefix_variance)
                #                                       .to(value_prefix_variance.device))
                # We compute the local uncertainty in the value of next state as the sum of next_state_based_uncertainty
                # (value_uncertainty) and previous-state-and-action uncertainty (value_prefix_uncertainty).
                if value_prefix_variance is not None and value_variance is not None:
                    # value_variance = value_variance + value_prefix_variance
                    value_variance = value_prefix_variance

                # Case UBE with either
                if 'ube' in self.uncertainty_type:
                    # We compute the value uncertainty as the sum of ube prediction (reliable on known states) and
                    # scaled value_uncertainty (more-reliable-than-UBE on unknown states).
                    if value_variance is not None:
                        assert (torch.is_tensor(value_variance) and len(value_variance.shape) == 1 and
                               value_variance.shape[0] == batch_size), f"type(value_variance) = {type(value_variance)} " \
                                                                      f"and expected flat tensor of size batch_size = " \
                                                                      f"{batch_size}"
                        ube_prediction = self.compute_ube_uncertainty(next_state)
                        if self.categorical_ube or len(ube_prediction.shape) > 1:
                            ube_prediction = self.inverse_ube_transform(ube_prediction).squeeze(-1)
                            assert ube_prediction.shape == value_variance.shape, \
                                f"ube_prediction.shape = {ube_prediction.shape}, " \
                                f"value_variance.shape = {value_variance.shape}, and should be equal." \
                                f"Expecting ({next_state.shape[0]})"

                        value_variance = torch.maximum(self.value_uncertainty_propagation_scale * value_variance,
                                                       ube_prediction.abs())
                    else:
                        ube_prediction = self.compute_ube_uncertainty(next_state)
                        if self.categorical_ube or len(ube_prediction.shape) > 1:
                            ube_prediction = self.inverse_ube_transform(ube_prediction).squeeze(-1)
                        ube_prediction = ube_prediction.abs()
                        value_variance = ube_prediction
                    # Upper bound the propagated value and compute:
                    # certainty * ube_prediction + uncertainty * propagation scale
                    # value_variance = value_prefix_variance * self.value_uncertainty_propagation_scale + \
                    #                  (1 - value_prefix_variance) * ube_prediction.abs()

                return value_variance.cpu().numpy() if value_variance is not None else None, \
                    value_prefix_variance.cpu().numpy() if value_prefix_variance is not None else None

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)
        value_variance, value_prefix_variance = None, None
        if 'double_model_rnd' in self.uncertainty_type:
            rnd_hidden_state = self.rnd_representation(obs)
        else:
            rnd_hidden_state = None

        if not self.training:
            # MuExplore: Compute the variance of the value prediction, and set the variance of the value_prefix
            value_variance, value_prefix_variance = self.compute_uncertainty(next_state=state, value=value,
                                                                             next_rnd_state=rnd_hidden_state)

            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            actor_logit = actor_logit.detach().cpu().numpy()
            if rnd_hidden_state is not None:
                rnd_hidden_state = rnd_hidden_state.detach().cpu().numpy()
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                             torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy())
        else:
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size).to('cuda'),
                             torch.zeros(1, num, self.lstm_hidden_size).to('cuda'))

        return NetworkOutput(value, [0. for _ in range(num)], actor_logit, state, reward_hidden, value_variance,
                             value_prefix_variance, rnd_hidden_state)

    def recurrent_inference(self, hidden_state, reward_hidden, action) -> NetworkOutput:
        if 'double_model_rnd' in self.uncertainty_type and isinstance(hidden_state, tuple):
            current_rnd_hidden_state = hidden_state[1]
            hidden_state = hidden_state[0]
            next_rnd_hidden_state = self.rnd_dynamics(current_rnd_hidden_state, action)
        else:
            current_rnd_hidden_state = None
            next_rnd_hidden_state = None
        state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        actor_logit, value = self.prediction(state)
        value_variance, value_prefix_variance = None, None

        if not self.training:
            # MuExplore: Compute the variance of the value prediction, and set the variance of the value_prefix
            value_variance, value_prefix_variance = self.compute_uncertainty(previous_state=hidden_state,
                                                                             next_state=state,
                                                                             action=action, value=value,
                                                                             value_prefix=value_prefix,
                                                                             previous_rnd_state=current_rnd_hidden_state,
                                                                             next_rnd_state=next_rnd_hidden_state)

            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            value_prefix = self.inverse_reward_transform(value_prefix).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            reward_hidden = (reward_hidden[0].detach().cpu().numpy(), reward_hidden[1].detach().cpu().numpy())
            actor_logit = actor_logit.detach().cpu().numpy()
            if next_rnd_hidden_state is not None:
                next_rnd_hidden_state = next_rnd_hidden_state.detach().cpu().numpy()

        return NetworkOutput(value, value_prefix, actor_logit, state, reward_hidden, value_variance,
                             value_prefix_variance, next_rnd_hidden_state)

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
