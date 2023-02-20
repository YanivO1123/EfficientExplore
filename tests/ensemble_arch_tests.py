import numpy
import torch
from config import atari
from core import utils
import numpy as np
from core.config import DiscreteSupport


class Ensemble_Arch_Tests:
    def __init__(self):
        self.game_config = atari.AtariConfig()
        self.game_config.use_uncertainty_architecture = False
        self.game_config.ensemble_size = 2
        self.game_config.stacked_observations  = 1
        self.game_config.gray_scale = False
        self.game_config.p_mcts_num = 1
        self.game_config.num_unroll_steps = 1
        self.game_config.batch_size = 2
        self.game_config.value_support = DiscreteSupport(-5, 5, delta=1)
        self.game_config.reward_support = DiscreteSupport(-5, 5, delta=1)
        self.game_config.lstm_hidden_size = 35
        self.proj_hid = 64
        self.proj_out = 64
        self.pred_hid = 32
        self.pred_out = 64

        self.game_config.set_game("BreakoutNoFrameskip-v4")
        self.env = self.game_config.new_game()
        print(f"self.env.action_space_size = {self.env.action_space_size}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None # The model is initialized in run_tests or run_prior_tests

    def run_tests(self):
        # Setup model
        self.model = self.game_config.get_uniform_network()
        self.model.to(self.device)

        # Run tests
        # self.test_initial_inference()
        # self.test_recurrent_inference()
        # self.test_inverse_scalar_transform_ensemble()
        # self.test_value_loss_computation()
        # self.test_reward_loss_computation()

    def run_prior_tests(self):
        # Setup model
        self.game_config.use_network_prior = True
        self.model = self.game_config.get_uniform_network()
        self.model.to(self.device)

        # Run tests
        self.test_losses_with_prior()
        self.test_initial_and_recurrent_inference_with_prior()
        self.test_ensemble_variance_with_prior()
        self.test_inverse_scalar_transform_with_prior()

    def transform_observation_from_atari_to_model(self, observation):
        # Turn the observation into a numpy array
        observation = utils.str_to_arr(observation)
        # Turn it into the output of step_obs, length of list is index:index + self.stacked_observations
        observation = [observation]
        # turn into the input to prepare_observation_lst, which is a list of length p_mcts_num
        observation = [observation]
        # observation = torch.from_numpy(np.array(observation)).to(self.game_config.device)
        observation = utils.prepare_observation_lst(observation)
        observation = torch.from_numpy(observation).to(self.game_config.device).float() / 255.0
        return observation

    def test_initial_inference(self):
        """
            Test that initial_inference with ensemble works as expected
                1. The output of initial_inference is of the right shape (just 1 value)
                2. The output of prediction is of the right shape (list of length ensemble)
        """
        self.model.eval()
        observation = self.env.reset()
        observation = self.transform_observation_from_atari_to_model(observation)
        initial_inference = self.model.initial_inference(observation)
        print(f"self.model.training = {self.model.training}")
        # value, value_prefix, policy_logits, hidden_state, reward_hidden
        print(f"The output of initial_inference(observation) is: \n"
              f"np.shape(initial_inference.value) = {np.shape(initial_inference.value)} \n"
              f"np.shape(initial_inference.value_prefix) = {np.shape(initial_inference.value_prefix)} \n"
              f"np.shape(initial_inference.policy_logits) = {np.shape(initial_inference.policy_logits)} \n"
              )

        hidden_state = self.model.representation(observation)
        actor_logit, value = self.model.prediction(hidden_state)

        # assert self.game_config.ensemble_size == len(value)

    def test_recurrent_inference(self):
        """
            Test that recurrent inference with ensemble works as expected
                1. The output of recurrent_inference is of the right shape (just 1 value)
                2. The output of prediction is of the right shape (list of length ensemble)
        """
        self.model.eval()
        observation = self.env.reset()
        observation = self.transform_observation_from_atari_to_model(observation)
        value, value_prefix, actor_logit, hidden_state, reward_hidden, _, _ = self.model.initial_inference(observation)
        print(f"numpy.shape(hidden_state) = {numpy.shape(hidden_state)} \n"
              f"numpy.shape(reward_hidden) = {numpy.shape(reward_hidden)} \n"
              f"numpy.shape(value) = {numpy.shape(value)} \n"
              f"numpy.shape(value_prefix) = {numpy.shape(value_prefix)} \n"
              f"numpy.shape(actor_logit) = {numpy.shape(actor_logit)} \n"
              )
        action_batch = [1]    # 1 is the action, first [] is for number of envs, [] is for number of actions in series
        action_batch = torch.from_numpy(numpy.asarray(action_batch)).to(self.game_config.device).unsqueeze(-1).long()
        hidden_state = torch.from_numpy(np.asarray(hidden_state)).to(self.device).float()
        reward_hidden_c = torch.from_numpy(np.asarray(reward_hidden[0])).to(self.device).float()
        reward_hidden_h = torch.from_numpy(np.asarray(reward_hidden[1])).to(self.device).float()
        print(f"hidden_state.shape = {hidden_state.shape}")
        print(f"reward_hidden_c.shape = {reward_hidden_c.shape}")
        print(f"reward_hidden_h.shape = {reward_hidden_h.shape}")
        print(f"action_batch.shape = {action_batch.shape}")
        value, value_prefix, actor_logit, state, reward_hidden, value_variance, value_prefix_variance = self.model.recurrent_inference(hidden_state, (reward_hidden_c, reward_hidden_h), action_batch)
        reward_w_dist, reward_mean = self.model.dynamics_network.get_reward_mean()
        print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n"
              f"numpy.shape(reward_w_dist) = {numpy.shape(reward_w_dist)} \n"
              f"numpy.shape(reward_w_dist) = {numpy.shape(reward_mean)} \n"
              f"reward_mean = {reward_mean} \n"
              f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # value, value_prefix, policy_logits, hidden_state, reward_hidden
        print(f"The output of recurrent_inference(hidden_state, reward_hidden, action_batch) is: \n"
              f"recurrent_inference.value = {value} \n"
              f"recurrent_inference.value_prefix = {value_prefix} \n"
              f"recurrent_inference.policy_logits = {actor_logit} \n"
              )
        print(f"And shapes: \n"
              f"numpy.shape(value) = {numpy.shape(value)} \n"
              f"numpy.shape(value_prefix) = {numpy.shape(value_prefix)} \n"
              f"numpy.shape(actor_logit) = {numpy.shape(actor_logit)} \n"
              f"numpy.shape(state) = {numpy.shape(state)} \n"
              f"numpy.shape(reward_hidden[0]) = {numpy.shape(reward_hidden[0])} \n"
              f"numpy.shape(reward_hidden[1]) = {numpy.shape(reward_hidden[1])} \n"
              )
        # Now, without calling the inverse transform function:
        state, reward_hidden, value_prefix = self.model.dynamics(hidden_state, (reward_hidden_c, reward_hidden_h), action_batch)
        actor_logit, value = self.model.prediction(state)
        print(f"The shapes of the outputs directly from dynamics and prediction are: \n"
              f"state.shape = {state.shape} \n"
              f"reward_hidden[0].shape = {reward_hidden[0].shape} \n"
              f"reward_hidden[1].shape = {reward_hidden[1].shape} \n"
              f"numpy.shape(value_prefix) = {numpy.shape(value_prefix)} \n"
              f"numpy.shape(value_prefix[0]) = {numpy.shape(value_prefix[0])} \n"
              f"actor_logit.shape = {actor_logit.shape} \n"
              f"numpy.shape(value) = {numpy.shape(value)} \n"
              f"numpy.shape(value[0]) = {numpy.shape(value[0])} \n"
              )

        # assert self.game_config.ensemble_size == len(value)
        # assert self.game_config.ensemble_size == len(value_prefix)

    def test_inverse_scalar_transform_ensemble(self):
        """
            Test that the output of recurrent and initial inference is the same as taking the average of the outputs of
            model.dynamics and model.prediction
        """
        self.model.eval()
        _ = self.env.reset()
        observation, _, _, _ = self.env.step(0)
        observation = self.transform_observation_from_atari_to_model(observation)
        value, value_prefix, actor_logit, hidden_state, reward_hidden, _, _ = self.model.initial_inference(observation)
        print(f"The value from initial_inference is: {value} \n"
              f"And value prefix (which is set to 0s): {value_prefix} \n")
        action_batch = [1]
        action_batch = torch.from_numpy(numpy.asarray(action_batch)).to(self.game_config.device).unsqueeze(-1).long()
        hidden_state = torch.from_numpy(np.asarray(hidden_state)).to(self.device).float()
        reward_hidden_c = torch.from_numpy(np.asarray(reward_hidden[0])).to(self.device).float()
        reward_hidden_h = torch.from_numpy(np.asarray(reward_hidden[1])).to(self.device).float()
        value_recurrent_inf, value_prefix, actor_logit, state, reward_hidden, _, _ = self.model.recurrent_inference(
            hidden_state, (reward_hidden_c, reward_hidden_h), action_batch
        )
        print(f"The value from recurrent_inference is: {value_recurrent_inf} \n"
              f"And value prefix (which is not set to 0s anymore): {value_prefix} \n")
        state, reward_hidden, value_prefix = self.model.dynamics(hidden_state, (reward_hidden_c, reward_hidden_h),
                                                                 action_batch)
        actor_logit, value = self.model.prediction(state)

        print(f"The size-600 value is a list of len: {len(value)}, with items of shape: {value[0].shape} \n")

        value_1 = self.model.inverse_value_transform(value).detach().cpu().numpy()
        value_2 = torch.zeros(numpy.shape(value_1))
        for i in range(self.game_config.ensemble_size):
            value_2 += self.model.inverse_value_transform(value[i]).detach().cpu().numpy()

        value_2 = (value_2 / self.game_config.ensemble_size).detach().cpu().numpy()

        print(f"The values are: \n"
              f"value_prediction_1 = {value_1} \n"
              f"value_prediction_2 = {value_2} \n"
              f"value_recurrent_inf = {value_recurrent_inf} \n"
              )

        # assert value_1 == value_2 == value_recurrent_inf

    def test_value_loss_computation(self):
        """
            Test that the computation of the value loss is the sum of the independent scalar losses
        """
        # Setup the model to training mode, for losses computation
        self.model.train()
        print("Testing value loss computation")
        # Setup the targets
        target_value = torch.tensor([[17], [-2]]).to(self.game_config.device).float()
        print(f"target_value.shape = {target_value.shape}, and value: {target_value} \n")
        transformed_target_value = self.game_config.scalar_transform(target_value)
        print(f"transformed_target_value.shape = {transformed_target_value.shape}, and value = {transformed_target_value} \n")
        target_value_phi = self.game_config.value_phi(transformed_target_value)
        print(f"target_value_phi.shape = {target_value_phi.shape}")
        # Setup the observation
        _ = self.env.reset()
        observation, _, _, _ = self.env.step(0)
        observation = self.transform_observation_from_atari_to_model(observation)
        observation = torch.cat((observation, observation))

        # Setup the initial inference predictions
        value, value_prefix, actor_logit, hidden_state, reward_hidden, _, _ = self.model.initial_inference(observation)
        print("#######################")
        print(f"len(value) = {len(value)}, and value[0].shape = {value[0].shape} \n"
              f"Expecting {self.game_config.ensemble_size} and 2,{self.game_config.value_support.size}")
        target_value_phi = target_value_phi.squeeze(1)
        print(f"target_value_phi.shape = {target_value_phi.shape}")

        # Setup the first loss:
        value_loss_from_ensemble_function = self.game_config.scalar_value_loss(value, target_value_phi)
        # print(f"value_loss_from_ensemble_function = {value_loss_from_ensemble_function}")

        print(f"value_loss_from_ensemble_function.shape = {value_loss_from_ensemble_function.shape} \n "
              f"and value_loss_from_ensemble_function = {value_loss_from_ensemble_function}")

        # Setup the second loss:
        value_loss_from_ensemble_manually = torch.zeros(2).to(self.game_config.device).float()
        for i in range(self.game_config.ensemble_size):
            value_loss_from_ensemble_manually += self.game_config.scalar_value_loss(value[i], target_value_phi)
        value_loss_from_ensemble_manually /= self.game_config.ensemble_size
        print(f"value_loss_from_ensemble_manually.shape = {value_loss_from_ensemble_manually.shape} \n"
              f"and value_loss_from_ensemble_manually = {value_loss_from_ensemble_manually}")

        # assert torch.equal(value_loss_from_ensemble_manually, value_loss_from_ensemble_function)

    def test_reward_loss_computation(self):
        """
            Test that the computation of the reward loss is the sum of the independent scalar losses
        """
        # Setup the model to training mode, for losses computation
        print("Setup model.train()")
        self.model.train()

        # Setup the targets
        print("Setup the targets")
        target_reward_prefix = torch.tensor([[11], [-4]]).to(self.game_config.device).float()
        print(f"target_reward_prefix.shape = {target_reward_prefix.shape}, and value: {target_reward_prefix}")
        transformed_target_reward_prefix = self.game_config.scalar_transform(target_reward_prefix)
        print(
            f"transformed_target_reward_prefix.shape = {transformed_target_reward_prefix.shape}, and value = {transformed_target_reward_prefix}")
        target_reward_prefix_phi = self.game_config.value_phi(transformed_target_reward_prefix)
        print(f"target_reward_prefix_phi.shape = {target_reward_prefix_phi.shape}")

        # Setup the observation
        print("Setup the observation")
        _ = self.env.reset()
        observation, _, _, _ = self.env.step(0)
        observation = self.transform_observation_from_atari_to_model(observation)
        observation = torch.cat((observation, observation))

        # Setup the initial inference predictions
        print("Setup the initial_inference")
        value, value_prefix, actor_logit, hidden_state, reward_hidden, _, _ = self.model.initial_inference(observation)

        # Setup the recurrent inference predictions
        print("Setup the recurrent_inference inputs")
        action_batch = [[1], [1]]  # 1 is the action, first [] is for number of envs, [] is for number of actions in series
        action_batch = torch.from_numpy(numpy.asarray(action_batch)).to(self.game_config.device).long()#.unsqueeze(-1)
        print(f"hidden_state.shape = {hidden_state.shape}")
        print(f"reward_hidden[0].shape = {reward_hidden[0].shape}")
        print(f"reward_hidden[1].shape = {reward_hidden[0].shape}")
        print(f"action_batch.shape = {action_batch.shape}")

        print("Run the recurrent_inference")
        value, value_prefix, actor_logit, state, reward_hidden, _, _ = self.model.recurrent_inference(hidden_state, reward_hidden, action_batch)

        print(f"len(value_prefix) = {len(value_prefix)}, and value_prefix[0].shape = {value_prefix[0].shape} \n"
              f"Expecting {self.game_config.ensemble_size} and [2, {self.game_config.reward_support.size}]")

        print(f"This is the value prefix: {value_prefix}")
        print(f"And these are the targets: {target_reward_prefix_phi[:,0]}")
        prediction_softmax = torch.log_softmax(value_prefix[0], dim=1)
        print(f"This is the softmax over the prediction for the first value: {prediction_softmax}")
        manual_loss = -(torch.log_softmax(prediction_softmax, dim=1) * target_reward_prefix_phi[:,0]).sum(1)
        print(f"And this is the manually computed loss, for the first value: {manual_loss}")


        # Setup the first loss:
        print("Computing the first loss, from direct function")
        reward_loss_from_ensemble_function = self.game_config.scalar_reward_loss(value, target_reward_prefix_phi[:,0])

        print(f"reward_loss_from_ensemble_function.shape = {reward_loss_from_ensemble_function.shape} \n "
              f"and reward_loss_from_ensemble_function = {reward_loss_from_ensemble_function}")

        # Setup the second loss:
        print("Computing the second loss, manually")
        reward_loss_from_ensemble_manually = torch.zeros(2).to(self.game_config.device).float()
        for i in range(self.game_config.ensemble_size):
            reward_loss_from_ensemble_manually += self.game_config.scalar_reward_loss(value[i], target_reward_prefix_phi[:,0])
        reward_loss_from_ensemble_manually /= self.game_config.ensemble_size
        print(f"reward_loss_from_ensemble_manually.shape = {reward_loss_from_ensemble_manually.shape} \n"
              f"and reward_loss_from_ensemble_manually = {reward_loss_from_ensemble_manually}")

        print(f"Shapes of losses are: \n"
              f"reward_loss_from_ensemble_manually.shape = {reward_loss_from_ensemble_manually.shape} and \n"
              f"reward_loss_from_ensemble_function.shape = {reward_loss_from_ensemble_function.shape} \n"
              f"and expected batch size which is {self.game_config.batch_size}")

        print("Asserting that losses are the same")
        # assert torch.equal(reward_loss_from_ensemble_manually, reward_loss_from_ensemble_function)

    def test_ensemble_variance(self):
        """
            Tests that the ensemble variance works as expected:
                1. The computed variance is equal the expected variance
                2. The shapes of the tensors are correct
        """
        self.model.eval()
        observation = self.env.reset()
        observation = self.transform_observation_from_atari_to_model(observation)
        value, value_prefix, actor_logit, hidden_state, reward_hidden = self.model.initial_inference(observation)
        action_batch = [1]  # 1 is the action, first [] is for number of envs, [] is for number of actions in series
        action_batch = torch.from_numpy(numpy.asarray(action_batch)).to(self.game_config.device).unsqueeze(-1).long()
        hidden_state = torch.from_numpy(np.asarray(hidden_state)).to(self.device).float()
        reward_hidden_c = torch.from_numpy(np.asarray(reward_hidden[0])).to(self.device).float()
        reward_hidden_h = torch.from_numpy(np.asarray(reward_hidden[1])).to(self.device).float()
        # Now, without calling the inverse transform function:
        state, reward_hidden, value_prefix = self.model.dynamics(hidden_state, (reward_hidden_c, reward_hidden_h),
                                                                 action_batch)
        actor_logit, value = self.model.prediction(state)

        variance = self.model.ensemble_prediction_to_variance(value)

    def test_losses_with_prior(self):
        raise NotImplementedError

    def test_initial_and_recurrent_inference_with_prior(self):
        raise NotImplementedError

    def test_ensemble_variance_with_prior(self):
        raise NotImplementedError

    def test_inverse_scalar_transform_with_prior(self):
        raise NotImplementedError



def ensemble_prediction_to_variance(logits):
    assert isinstance(logits, list)
    print("In function ensemble_prediction_to_variance")
    print(f"Input is a list of len: {len(logits)}, of tensors of shape: {logits[0].shape}")
    print(f"Input logits = {logits}")
    with torch.no_grad():
        # Softmax the logits
        logits = [torch.softmax(logits[i], dim=1) for i in range(len(logits))]
        print(f"After softmax: logits is a list of len: {len(logits)}, of tensors of shape: {logits[0].shape}")
        print(f"After softmax: logits = {logits}")
        # Stack into a tensor to compute the variance using torch
        # shape should be: (ensemble_size, batch_size, full_support_size)
        stacked_tensor = torch.stack(logits, dim=0)
        print(f"After stacking, stacked_tensor.shape = {stacked_tensor.shape}, and expected: (4,2,3)")
        print(f"After stacking: stacked_tensor = {stacked_tensor}")
        # Compute the per-entry variance over the dimension 0 (ensemble size)
        scalar_variance = torch.var(stacked_tensor, unbiased=False, dim=0)
        print(f"After computing variances, scalar_variance.shape = {scalar_variance.shape}, and expected: (2,3)")
        print(f"And the scalar_variance tensor itself: {scalar_variance}, expected: ([const, 0, 0], [const, 0, 0])")
        # Sum the per-entry variance scores
        scalar_variance = scalar_variance.sum(-1)
        print(f"After summing variances, scalar_variance.shape = {scalar_variance.shape}, and expected: (2)")
        print(f"And the scalar_variance tensor itself: {scalar_variance}, expected: ([const, const])")
        # Shape should be: (batch_size, 1)
        return scalar_variance

def test_ensemble_prediction_to_variance():
    # shape: list of size ensemble size, of tensors of shape: (parl_envs, full_support)
    ensemble_size = 4
    parl_envs = 2
    full_support = 3
    logits = [torch.from_numpy(numpy.asarray([[2.0, 2.0, 2.0], [0.8, 0.1, 0.1]])).float(),
              torch.from_numpy(numpy.asarray([[2.0, 2.0, 2.0], [0.8, 0.1, 0.1]])).float(),
              torch.from_numpy(numpy.asarray([[2.0, 2.0, 2.0], [0.7, 0.2, 0.1]])).float(),
              torch.from_numpy(numpy.asarray([[2.0, 2.0, 2.0], [0.7, 0.2, 0.1]])).float()]
    computed_variance = ensemble_prediction_to_variance(logits)
    first_entry_variance = np.var([0.5017, 0.5017, 0.4640, 0.4640])
    second_entry_variance = np.var([0.2491, 0.2491, 0.2814, 0.2814])
    third_entry_variance = np.var([0.2491, 0.2491, 0.2546, 0.2546])
    print(f"Individual vars. computed with numpy: \n"
          f"first_entry_variance = {first_entry_variance} \n"
          f"second_entry_variance = {second_entry_variance} \n"
          f"third_entry_variance = {third_entry_variance} \n"
          f"And sum of all three = {first_entry_variance + second_entry_variance + third_entry_variance} \n"
          )

# test_ensemble_prediction_to_variance()
# Init and run the testing framework
Ensemble_Arch_Tests().run_tests()