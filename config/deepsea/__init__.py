import torch
from bsuite import sweep
from core.config import BaseConfig
from core.utils import make_deepsea, EpisodicLifeEnv
from core.dataset import Transforms
from config.deepsea.env_wrapper import DeepSeaWrapper
from config.deepsea.model import EfficientZeroNet
from config.deepsea.model import EfficientExploreNet
from core.config import DiscreteSupport
import bsuite
from bsuite.utils import gym_wrapper


class DeepSeaConfig(BaseConfig):
    def __init__(self):
        super(DeepSeaConfig, self).__init__(
            training_steps=100000, #100000,
            last_steps=20000,#20000
            test_interval=500, #10000,
            log_interval=500,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=10, #12000, # Max moves are re-set in set_game to env_size
            test_max_moves=10, #12000, # test_max_moves are re-set in set_game to env_size
            history_length=10, # history_length is re-set in set_game to env_size
            discount=0.997, # Might want lower
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=64, # 32 # 64 #256,  # TODO: can be larger with smaller net
            td_steps=10, # 5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=False, # This uses properties of real gym
            init_zero=True,
            clip_reward=False,  # no need to clip reward in deepsea
            # storage efficient
            cvt_string=False,    # deepsea returns a 1 hot encoding, no need to
            image_based=False,   #
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=50000,
            auto_td_steps_ratio=0.1, # 0.3,
            # replay window
            start_transitions=400,
            total_transitions=100 * 1000,
            transition_num=1,
            do_consistency=True,
            # frame skip & stack observation
            frame_skip=1,      # TODO: I believe this is skipping * 1
            stacked_observations=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            lstm_hidden_size=128, #512,    # TODO: Can lower aggressively
            lstm_horizon_len=5,
            # siamese
            proj_hid=64, #1024,    # TODO: Can lower aggressively, and also check relevance with deepsea observations
            proj_out=64, #1024,    # TODO: Can lower aggressively, and also check relevance with deepsea observations
            pred_hid=32, #512,     # TODO: Can lower aggressively, and also check relevance with deepsea observations
            pred_out=64, #1024,    # TODO: Can lower aggressively, and also check relevance with deepsea observations
            value_support=DiscreteSupport(-15, 15, delta=1),
            reward_support=DiscreteSupport(-15, 15, delta=1),
            # MuExplore
            # Architecture
            use_uncertainty_architecture=False,
            ensemble_size=5,
            use_network_prior=True,
            prior_scale=10.0,
            # visitation counter
            use_visitation_counter=False,
            plan_with_visitation_counter=False,
            # Exploration
            mu_explore=False,
            beta=1.0,
            disable_policy_in_exploration=False,
            # ratio of training / interactions
            training_ratio=1,
        )
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 8 # 64 # Number of channels in the ResNet
        self.reduced_channels_reward = 8 # 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 8 # 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 8 # 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32, 32] # [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32, 32] # [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32] # [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
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
        env_size = sweep.SETTINGS[env_name]['size']
        self.image_channel = 1
        obs_shape = (self.image_channel, env_size, env_size)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])
        self.history_length = env_size
        self.max_moves = env_size
        self.test_max_moves = env_size
        game = self.new_game()

        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        if self.use_uncertainty_architecture:
            return EfficientExploreNet(
                self.obs_shape,
                self.action_space_size,
                self.blocks,
                self.channels,
                self.reduced_channels_reward,
                self.reduced_channels_value,
                self.reduced_channels_policy,
                self.resnet_fc_reward_layers,
                self.resnet_fc_value_layers,
                self.resnet_fc_policy_layers,
                self.reward_support.size,
                self.value_support.size,
                self.downsample,
                self.inverse_value_transform,
                self.inverse_reward_transform,
                self.lstm_hidden_size,
                bn_mt=self.bn_mt,
                proj_hid=self.proj_hid,
                proj_out=self.proj_out,
                pred_hid=self.pred_hid,
                pred_out=self.pred_out,
                init_zero=self.init_zero,
                state_norm=self.state_norm,
                ensemble_size=self.ensemble_size,
                use_network_prior=self.use_network_prior,
                prior_scale=self.prior_scale
            )
        else:
            return EfficientZeroNet(
                self.obs_shape,
                self.action_space_size,
                self.blocks,
                self.channels,
                self.reduced_channels_reward,
                self.reduced_channels_value,
                self.reduced_channels_policy,
                self.resnet_fc_reward_layers,
                self.resnet_fc_value_layers,
                self.resnet_fc_policy_layers,
                self.reward_support.size,
                self.value_support.size,
                self.downsample,
                self.inverse_value_transform,
                self.inverse_reward_transform,
                self.lstm_hidden_size,
                bn_mt=self.bn_mt,
                proj_hid=self.proj_hid,
                proj_out=self.proj_out,
                pred_hid=self.pred_hid,
                pred_out=self.pred_out,
                init_zero=self.init_zero,
                state_norm=self.state_norm)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False,
                 final_test=False):
        env = make_deepsea(self.env_name, seed=self.seed)

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

    # TODO:
    def ube_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

game_config = DeepSeaConfig()
