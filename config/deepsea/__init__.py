import torch
from bsuite import sweep
from core.config import BaseConfig
from core.utils import make_deepsea, EpisodicLifeEnv
from core.dataset import Transforms
from config.deepsea.env_wrapper import DeepSeaWrapper
from config.deepsea.model import FullyConnectedEfficientExploreNet
from core.config import DiscreteSupport
import bsuite
from bsuite.utils import gym_wrapper
from core.utils import init_kaiming_trunc_haiku


class DeepSeaConfig(BaseConfig):
    def __init__(self):
        super(DeepSeaConfig, self).__init__(
            training_steps=50 * 1000, #100000,
            last_steps=0,#20000
            test_interval=100, #10000, 500
            log_interval=100,
            vis_interval=100,   # 1000
            test_episodes=8, # 32,
            checkpoint_interval=100,    # 100
            target_model_interval=200,  # 200
            save_ckpt_interval=10000,
            max_moves=10,   # Max moves are re-set in set_game to env_size
            test_max_moves=10,  # test_max_moves are re-set in set_game to env_size
            history_length=10,  # history_length is re-set in set_game to env_size
            discount=0.995,     # 0.98, 0.997
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=64,  # 32 # 64 #256,  # TODO: can be larger with smaller net
            td_steps=5,     # 5, 10, 3, 1
            num_actors=1,
            # network initialization/ & normalization
            episode_life=False, # This uses properties of real gym
            init_zero=False,
            clip_reward=False,  # no need to clip reward in deepsea
            # storage efficient
            cvt_string=False,    # deepsea returns a 1 hot encoding, no need to
            image_based=False,   #
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=1E-3,    # torch_amp=0.2, none=1E-3,
            lr_decay_rate=0.1,     # 0.1
            lr_decay_steps=50 * 1000,
            num_unroll_steps=5, # 5, 10    The hardcoded default is 5. Might not work reliably with other values
            auto_td_steps_ratio=0.3,    # 0.3, 0.1
            # replay window
            start_transitions=64,   # 500 400 32 5000 1000
            total_transitions=50 * 1000,
            transition_num=1,
            do_consistency=True,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=1,     # 4 2
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.5,  # 0.25 original # 1 0.5
            policy_loss_coeff=0.5,
            consistency_coeff=4,
            ube_loss_coeff=0.25,
            # reward sum
            lstm_hidden_size=64, #512
            use_value_prefix=False,
            lstm_horizon_len=1,
            # siamese
            proj_hid=512, # 64, 1024
            proj_out=512, # 64, 1024
            pred_hid=256, # 32, 512
            pred_out=512, # 64, 1024,
            value_support=DiscreteSupport(-10, 10, delta=1),    # worked with 10
            reward_support=DiscreteSupport(-10, 10, delta=1),   # worked with 10
            # MuExplore
            # Architecture
            ensemble_size=5,
            use_prior=True,
            prior_scale=5.0,
            # visitation counter
            use_visitation_counter=False,
            plan_with_visitation_counter=False,
            # Exploration
            mu_explore=False,
            beta=1.0,
            # ratio of training / interactions
            training_ratio=1,
            # UBE params
            ube_td_steps=2,
            reset_ube_interval=5000,
            rnd_scale=0.1,
            ube_support=DiscreteSupport(0, 10, delta=1),
        )
        self.start_transitions = max(1, self.start_transitions)

        # Architecture specification
        self.bn_mt = 0.1

        # Fullyconnected arch. specs
        self.identity_representation = True
        self.fc_representation_layers = [128, 128]
        self.fc_state_prediction_layers = [128, 128, 128] # [64] [128, 128, 128]
        self.fc_state_prediction_prior_layers = [128, 128, 128] # [128, 128, 128]
        self.fc_reward_layers = [128, 128] # [64, 64], [128, 128]
        self.fc_reward_prior_layers = [256, 128]    # [128, 128]
        self.fc_value_layers = [128, 128]           # [64, 64], [128, 128]
        self.fc_value_prior_layers = [256, 128]     # [128, 128]
        self.fc_policy_layers = [128, 128]          # [64, 64], [128, 128]
        # UBE architecture
        self.fc_ube_layers = [128, 128, 128]
        self.categorical_ube = True

        # RND architecture
        self.fc_rnd_layers = [1024, 256]
        self.fc_rnd_target_layers = [1024, 1024, 256]
        self.fc_lstm_hidden_size = self.lstm_hidden_size
        # Encoder architecture
        self.use_encoder = False
        self.encoder_layers = [1024, 1024, 256]
        self.encoding_size = 4  # The encoded state is of size self.encoding_size * self.encoding_size

        self.reset_all_weights = False

        # To reduce the effect of the policy on the selection, we reduce pb_c_init to 0.5.
        # This should give the policy about half the weight
        # self.pb_c_init = 0.5 # 1.25

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        # With mu explore in deep sea, we don't want to rely on random action selection
        if self.mu_explore:
            return 0.1
        elif self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None, dimensions=9):
        self.env_name = env_name
        self.env_size = sweep.SETTINGS[env_name]['size']
        self.image_channel = 1
        obs_shape = (self.image_channel, self.env_size, self.env_size)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])
        self.history_length = self.env_size
        self.max_moves = self.env_size
        self.test_max_moves = self.env_size
        game = self.new_game()

        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        model = FullyConnectedEfficientExploreNet(
            self.obs_shape,
            self.action_space_size,
            self.identity_representation,
            self.fc_representation_layers,
            self.fc_state_prediction_layers,
            self.fc_state_prediction_prior_layers,
            self.fc_reward_layers,
            self.fc_reward_prior_layers,
            self.fc_value_layers,
            self.fc_value_prior_layers,
            self.fc_policy_layers,
            self.fc_rnd_layers,
            self.fc_rnd_target_layers,
            self.fc_ube_layers,
            self.value_support.size,
            self.reward_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.fc_lstm_hidden_size,
            momentum=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            rnd_scale=self.rnd_scale,
            learned_model=self.learned_model,
            mapping_seed=self.seed,
            randomize_actions=self.deepsea_randomize_actions,
            uncertainty_type=self.uncertainty_architecture_type,
            discount=self.discount,
            ensemble_size=self.ensemble_size,
            use_prior=self.use_prior,
            prior_scale=self.prior_scale,
            use_encoder=self.use_encoder,
            encoder_layers=self.encoder_layers,
            encoding_size=self.encoding_size,
            categorical_ube=self.categorical_ube,
            inverse_ube_transform=self.inverse_ube_transform,
            ube_support_size=self.ube_support.size,
        )

        if 'rnd' in self.uncertainty_architecture_type:
            for p in model.value_rnd_target_network.parameters():
                p.requires_grad = False
                p *= 4
            for p in model.reward_rnd_target_network.parameters():
                p.requires_grad = False
                p *= 4
        if 'ensemble' in self.uncertainty_architecture_type:
            model.dynamics_network.fc.apply(fn=init_kaiming_trunc_haiku)
            model.value_network.apply(fn=init_kaiming_trunc_haiku)
            if self.use_prior:
                model.dynamics_network.fc_net_prior.apply(fn=init_kaiming_trunc_haiku)
                model.value_network_prior.apply(fn=init_kaiming_trunc_haiku)
                for p in model.dynamics_network.fc_net_prior.parameters():
                    p.requires_grad = False
                    p *= 2
                for p in model.value_network_prior.parameters():
                    p.requires_grad = False
                    p *= 2
        if self.use_encoder:
            model.representation_encoder.apply(fn=init_kaiming_trunc_haiku)
            for p in model.representation_encoder.parameters():
                p.requires_grad = False
                p *= 2

        return model

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False,
                 final_test=False):
        # We make all deep_sea envs with the same action mapping seed, which is the seed of the config
        # The input seed can be used to init the rest of the env (for stoch. deepsea)
        env = make_deepsea(self.env_name, seed=self.seed, randomize_actions=self.deepsea_randomize_actions)

        if save_video:
            print("Does not have save_video option in deep_sea, proceeding without")

        return DeepSeaWrapper(env, discount=self.discount, cvt_string=False)

    def scalar_reward_loss(self, prediction, target):
        if isinstance(prediction, list):
            return self.ensemble_scalar_reward_loss(prediction, target)
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        if isinstance(prediction, list):
            return self.ensemble_scalar_value_loss(prediction, target)
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)

    #MuExplore: Compute the loss over an ensemble
    def ensemble_scalar_reward_loss(self, predictions, target):
        assert type(predictions) is list
        scalar_loss_list = [self.scalar_reward_loss(prediction, target) for prediction in predictions]
        scalar_loss_tensor = torch.stack(scalar_loss_list, dim=0)
        # shape of scalar_loss_tensor should be: (ensemble_size, batch_size, 1)
        scalar_loss = torch.mean(scalar_loss_tensor, dim=0)
        return scalar_loss

    #MuExplore: Compute the loss over an ensemble
    def ensemble_scalar_value_loss(self, predictions, target):
        assert type(predictions) is list
        scalar_loss_list = [self.scalar_value_loss(prediction, target) for prediction in predictions]
        scalar_loss_tensor = torch.stack(scalar_loss_list, dim=0)
        # shape of scalar_loss_tensor should be: (ensemble_size, batch_size, 1)
        scalar_loss = torch.mean(scalar_loss_tensor, dim=0)
        return scalar_loss

    def ube_loss(self, prediction, target):
        if self.categorical_ube:
            return -(torch.log_softmax(prediction, dim=1) * target).sum(1)
        else:
            return torch.nn.functional.mse_loss(prediction, target, reduction='none')

    def inverse_ube_transform(self, ube_logits):
        return self.inverse_scalar_transform(ube_logits, self.ube_support)

    def ube_phi(self, x):
        return self._phi(x, self.ube_support.min, self.ube_support.max, self.ube_support.size)

    # def rnd_loss(self, prediction, target):
    #     return torch.nn.functional.mse_loss(prediction, target, reduction='none')

game_config = DeepSeaConfig()
