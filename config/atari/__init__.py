import torch

from core.config import BaseConfig
from core.utils import make_atari, WarpFrame, EpisodicLifeEnv
from core.dataset import Transforms
from .env_wrapper import AtariWrapper
from .model import EfficientZeroNet
from .model import EfficientExploreNet


class AtariConfig(BaseConfig):
    def __init__(self):
        super(AtariConfig, self).__init__(
            training_steps=100000,
            last_steps=20000,
            test_interval=5000, #10000,
            log_interval=1000,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=12000,
            test_max_moves=12000,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=32, # 32 # 64 #256,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=True,
            # storage efficient
            cvt_string=True,
            image_based=True,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=100000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=8,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=4,
            stacked_observations=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            lstm_hidden_size=256, #512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=256, #1024,
            proj_out=256, #1024,
            pred_hid=128, #512,
            pred_out=256, #1024,
            # MuExplore
            # Architecture
            use_uncertainty_architecture=True,
            ensemble_size=2,
            use_network_prior=True,
            prior_scale=10.0,
            # Exploration
            mu_explore=True,
            beta=0,
            disable_policy_in_exploration = False,
            # ratio of training / interactions
            training_ratio = 1,
        )
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32, 32] # [32] # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32, 32] # [32] # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_rnd_layers = [1024, 1024, 1024, 256]  # The last number is interpreted as outputsize
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)

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

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        # gray scale
        if self.gray_scale:
            self.image_channel = 1
        obs_shape = (self.image_channel, 96, 96)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        if self.use_uncertainty_architecture:
            print("Initiating EfficientExploreNet")
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
                self.resnet_fc_rnd_layers,
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

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.test_max_moves
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        else:
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=self.max_moves)

        if self.episode_life and not test:
            env = EpisodicLifeEnv(env)
        env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return AtariWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

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

game_config = AtariConfig()
