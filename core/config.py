import os
import time
import torch

import numpy as np

from core.game import Game


class DiscreteSupport(object):
    def __init__(self, min: int, max: int, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1, delta)
        self.size = len(self.range)
        self.delta = delta


class BaseConfig(object):

    def __init__(self,
                 training_steps: int,
                 last_steps: int,
                 test_interval: int,
                 test_episodes: int,
                 checkpoint_interval: int,
                 target_model_interval: int,
                 save_ckpt_interval: int,
                 log_interval: int,
                 vis_interval: int,
                 max_moves: int,
                 test_max_moves: int,
                 history_length: int,
                 discount: float,
                 dirichlet_alpha: float,
                 value_delta_max: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_warm_up: float,
                 lr_init: float,
                 lr_decay_rate: float,
                 lr_decay_steps: float,
                 start_transitions: int,
                 num_unroll_steps: int = 5,
                 auto_td_steps_ratio: float = 0.3,
                 total_transitions: int = 100 * 1000,
                 transition_num: float = 25,
                 do_consistency: bool = True,
                 use_value_prefix: bool = True,
                 off_correction: bool = True,
                 gray_scale: bool = False,
                 episode_life: bool = False,
                 change_temperature: bool = True,
                 init_zero: bool = False,
                 state_norm: bool = False,
                 clip_reward: bool = False,
                 random_start: bool = True,
                 cvt_string: bool = False,
                 image_based: bool = False,
                 frame_skip: int = 1,
                 stacked_observations: int = 16,
                 lstm_hidden_size: int = 64,
                 lstm_horizon_len: int = 1,
                 reward_loss_coeff: float = 1,
                 value_loss_coeff: float = 1,
                 policy_loss_coeff: float = 1,
                 consistency_coeff: float = 1,
                 proj_hid: int = 256,
                 proj_out: int = 256,
                 pred_hid: int = 64,
                 pred_out: int = 256,
                 value_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1),
                 reward_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1),
                 mu_explore: bool = False,
                 use_visitation_counter: bool = False,
                 plan_with_visitation_counter: bool = False,
                 ensemble_size: int = 5,
                 use_prior: bool = False,
                 prior_scale: float = 10.0,
                 beta: float = 1,
                 disable_policy_in_exploration: bool = True,
                 training_ratio: float = 1,
                 use_max_value_targets: bool = False,
                 use_max_policy_targets: bool = False,
                 sampling_times: int = 0,
                 ube_td_steps: int = 5,
                 ube_loss_coeff: float = 2,
                 ube_support: DiscreteSupport = DiscreteSupport(0, 300, delta=1),
                 count_based_ube: bool = False,
                 num_simulations_ube: int = 30,
                 reset_ube_interval: int = 1000 * 5,
                 rnd_scale: float = 1.0,
                 ):
        """Base Config for EfficietnZero
        Parameters
        ----------
        training_steps: int
            training steps while collecting data
        last_steps: int
            training steps without collecting data after @training_steps;
            So total training steps = training_steps + last_steps
        test_interval: int
            interval of testing
        test_episodes: int
            episodes of testing
        checkpoint_interval: int
            interval of updating the models for self-play
        target_model_interval: int
            interval of updating the target models for reanalyzing
        save_ckpt_interval: int
            interval of saving models
        log_interval: int
            interval of logging
        vis_interval: int
            interval of visualizations for some distributions and loggings
        max_moves: int
            max number of moves for an episode
        test_max_moves: int
            max number of moves for an episode during testing (in training stage),
            set this small to make sure the game will end faster.
        history_length: int
            horizons of each stored history trajectory.
            The horizons of Atari games are quite large. Split the whole trajectory into several history blocks.
        discount: float
            discount of env
        dirichlet_alpha: float
            dirichlet alpha of exploration noise in MCTS.
            Smaller -> more exploration
        value_delta_max: float
            the threshold in the minmax normalization of Q-values in MCTS.
            See the soft minimum-maximum updates in Appendix.
        num_simulations: int
            number of simulations in MCTS
        batch_size: int,
            batch size
        td_steps: int
            td steps for bootstrapped value targets
        num_actors: int
            number of self-play actors
        lr_warm_up: float
            rate of learning rate warm up
        lr_init: float
            initial lr
        lr_decay_rate: float
            how much lr drops every time
            lr -> lr * lr_decay_rate
        lr_decay_steps: float
            lr drops every lr_decay_steps
        start_transitions: int
            least transition numbers to start the training steps ( larger than batch size)
        num_unroll_steps: int
            The number of steps to unroll in training the recurrent networks of MuZero
        auto_td_steps_ratio: float
            ratio of short td steps, samller td steps for older trajectories.
            auto_td_steps = auto_td_steps_ratio * training_steps
            See the details of off-policy correction in Appendix.
        total_transitions: int
            total number of collected transitions. (100k setting)
        transition_num: float
            capacity of transitions in replay buffer
        do_consistency: bool
            True -> use temporal consistency
        use_value_prefix: bool = True,
            True -> predict value prefix
        off_correction: bool
            True -> use off-policy correction
        gray_scale: bool
            True -> use gray image observation
        episode_life: bool
            True -> one life in atari games
        change_temperature: bool
            True -> change temperature of visit count distributions
        init_zero: bool
            True -> zero initialization for the last layer of mlps
        state_norm: bool
            True -> normalization for hidden states
        clip_reward: bool
            True -> clip the reward, reward -> sign(reward)
        random_start: bool
            True -> random actions in self-play before startng training
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        image_based: bool
            True -> observation is image based
        frame_skip: int
            number of frame skip
        stacked_observations: int
            number of frame stack
        lstm_hidden_size: int
            dim of lstm hidden
        lstm_horizon_len: int
            horizons of value prefix prediction, 1 <= lstm_horizon_len <= num_unroll_steps
        reward_loss_coeff: float
            coefficient of reward loss
        value_loss_coeff: float
            coefficient of value loss
        policy_loss_coeff: float
            coefficient of policy loss
        consistency_coeff: float
            coefficient of consistency loss
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        value_support: DiscreteSupport
            support of value to represent the value scalars
        reward_support: DiscreteSupport
            support of reward to represent the reward scalars
        mu_explore: bool
            whether to use mu_explore in MCTS (i.e. out of p_mcts_num envs, one is exploitatory (standard), and the rest
            are exploratory (with MuExplore)
        use_visitation_counter: bool
            Only implemented for deep_sea. The visitation counter counts state-action visitations and uses them as a
            measure of epistemic uncertainty, as so: 1 / (epsilon + state_action_visit_count)
        plan_with_visitation_counter: bool
            Whether to use the visitation counter in planning with MCTS and muexplore, or only for debugging purposes
        ensemble_size: int
            The number of ensemble-heads, both for value prediction as well as value prefix prediction.
        use_prior: bool
            Whether to use prior functions on the ensemble heads, or not
        prior_scale: float
            The scale of the prior in the computation
        beta: float
            used in MuExplore in UCB computation to prioritize exploration / exploitation, as follows:
                MuExplore_UCB = value + beta * sqrt(value_variance)
        disable_policy_in_exploration: bool
            If True, the agent does not use the policy to compute the prior in exploration episodes.
            This is important because the policy heavily influences MCTS search and thus action selection, and in
            exploration the agent should be free to explore in varied directions.
        training_ratio: float
            a float >= 1, guarantees that ratio: % trained steps / % transitions >= training_ratio
            The purpose is to encourage / require / enable the agent to train much more than interact.
        use_max_value_targets: bool
            If True, uses max_value_targets (see paper, learning from exploratory decisions). Otherwise, does standard
            EfficientZero.
        use_max_policy_targets: bool
            If True, uses max_policy_targets from max_value_targets (see paper, learning from exploratory decisions). Otherwise, does standard
            EfficientZero.
        sampling_times: int
            If using the visitataion counter, and sampled value uncertainty propagation, how many times to sample.
        ube_td_steps: int
            td steps for bootstrapped value-uncertainty targets for UBE
        ube_loss_coeff: float
            coefficient of ube loss. ube gradients do not flow into the model. However, we want UBE to learn "faster"
            than the value function, so we use coefficient that is larger than value coeff
        ube_support: DiscreteSupport
            support of ube to represent the ube scalars. Should be between 0 and max value
        count_based_ube: bool
            Whether to use visitation counts as target for UBE. Defaults to false.
        num_simulations_ube: int
            The number of simulations to use to compute ube targets in reanalyze when root_value is used.
        reset_ube_interval: int
            When step_count % config.reset_ube_interval == 0 reset the network parameters of UBE, to prevent
            failure to adapt to new uncertainty scores.
            If reset_ube_interval <= 0, don't reset ube weights.
        rnd_scale: float
            In compute_rnd_uncertainty functions, scales the MSE error as a measure of uncertainty.
        """
        # Self-Play
        self.action_space_size = None
        self.num_actors = num_actors
        self.do_consistency = do_consistency
        self.use_value_prefix = use_value_prefix
        self.off_correction = off_correction
        self.gray_scale = gray_scale
        self.auto_td_steps_ratio = auto_td_steps_ratio
        self.episode_life = episode_life
        self.change_temperature = change_temperature
        self.init_zero = init_zero
        self.state_norm = state_norm
        self.clip_reward = clip_reward
        self.random_start = random_start
        self.cvt_string = cvt_string
        self.image_based = image_based

        self.max_moves = max_moves
        self.test_max_moves = test_max_moves
        self.history_length = history_length
        self.num_simulations = num_simulations
        self.discount = discount
        self.max_grad_norm = 5

        # testing arguments
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        # Root prior exploration noise.
        self.value_delta_max = value_delta_max
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25   # original: 0.25, first attempt reduced: 0.1

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.training_steps = training_steps
        self.last_steps = last_steps
        self.checkpoint_interval = checkpoint_interval
        self.target_model_interval = target_model_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.log_interval = log_interval
        self.vis_interval = vis_interval
        self.start_transitions = start_transitions
        self.total_transitions = total_transitions
        self.transition_num = transition_num
        self.batch_size = batch_size
        # unroll steps
        self.num_unroll_steps = num_unroll_steps # 5
        self.td_steps = td_steps
        self.frame_skip = frame_skip
        self.stacked_observations = stacked_observations
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_horizon_len = lstm_horizon_len
        self.reward_loss_coeff = reward_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.consistency_coeff = consistency_coeff
        self.device = 'cuda'
        self.exp_path = None  # experiment path
        self.debug = False
        self.model_path = None
        self.seed = None
        self.transforms = None
        self.value_support = value_support
        self.reward_support = reward_support

        # optimization control
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_warm_up = lr_warm_up
        self.lr_warm_step = int(self.training_steps * self.lr_warm_up)
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.mini_infer_size = 64

        # replay buffer, priority related
        self.priority_prob_alpha = 0.6
        self.priority_prob_beta = 0.4
        self.prioritized_replay_eps = 1e-6

        # env
        self.image_channel = 3

        # contrastive arch
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out

        ## uncertainty
        # architecture
        self.architecture_type = 'resnet'   # Decides the architecture of the NN. 'fully_connected' is only implemented for deepsea
        self.use_uncertainty_architecture = False
        self.uncertainty_architecture_type = None   # default is set in main, in parameter parsing

        # ensemble
        self.ensemble_size = ensemble_size
        self.use_prior = use_prior
        self.prior_scale = prior_scale

        # rnd
        self.rnd_scale = rnd_scale

        # exploration
        self.use_deep_exploration = False       # Activated in set_config if mu_explore or ube
        self.mu_explore = mu_explore
        self.beta = beta
        self.disable_policy_in_exploration = disable_policy_in_exploration
        self.number_of_exploratory_envs = 0     # Setup in set_config if self.use_deep_exploration

        # visitation counter
        self.use_visitation_counter = use_visitation_counter
        self.plan_with_visitation_counter = plan_with_visitation_counter
        self.plan_with_state_visits = False
        self.plan_with_fake_visit_counter = False

        # control training / interactions ratio
        self.training_ratio = training_ratio

        # Target types in exploration
        self.use_max_value_targets = use_max_value_targets
        self.use_max_policy_targets = use_max_policy_targets

        # Action selection:
        self.standard_action_selection = True   # Set to false in set_config if not MuExplore and yes ube

        # UBE hyper parameters
        self.ube_td_steps = ube_td_steps
        self.ube_loss_coeff = ube_loss_coeff
        self.count_based_ube = count_based_ube
        self.num_simulations_ube = num_simulations_ube
        self.reset_ube_interval = reset_ube_interval
        self.periodic_ube_weight_reset = False
        self.reset_all_weights = False
        self.ube_support = ube_support

        # Deep sea for debugging
        self.deepsea_randomize_actions = True
        self.learned_model = True
        self.env_size = -1  # reset in set_game
        self.sampling_times = sampling_times
        self.use_encoder = False
        self.representation_based_training = False

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        raise NotImplementedError

    def set_game(self, env_name):
        raise NotImplementedError

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False) -> Game:
        """ returns a new instance of the game"""
        raise NotImplementedError

    def get_uniform_network(self):
        raise NotImplementedError

    def scalar_loss(self, prediction, target):
        raise NotImplementedError

    def scalar_transform(self, x):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        delta = self.value_support.delta
        assert delta == 1
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    def inverse_scalar_transform(self, logits, scalar_support):
        """ Reference from MuZerp: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        # If it's an ensemble output, return the mean prediction
        if isinstance(logits, list):
            mean_value = self.inverse_scalar_transform(logits[0], scalar_support)
            for logits_index in range(1, len(logits)):
                mean_value += self.inverse_scalar_transform(logits[logits_index], scalar_support)
            mean_value = mean_value / len(logits)
            return mean_value
        else:
            delta = self.value_support.delta
            value_probs = torch.softmax(logits, dim=1)
            value_support = torch.ones(value_probs.shape)
            value_support[:, :] = torch.from_numpy(np.array([x for x in scalar_support.range]))
            value_support = value_support.to(device=value_probs.device)
            value = (value_support * value_probs).sum(1, keepdim=True) / delta

            epsilon = 0.001
            sign = torch.ones(value.shape).float().to(value.device)
            sign[value < 0] = -1.0
            output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
            output = sign * output * delta

            nan_part = torch.isnan(output)
            output[nan_part] = 0.
            output[torch.abs(output) < epsilon] = 0.
            return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size: int):
        delta = self.value_support.delta

        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    def get_hparams(self):
        # get all the hyper-parameters
        hparams = {}
        for k, v in self.__dict__.items():
            if 'path' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        # reset config from the args
        self.set_game(args.env)
        self.case = args.case
        self.seed = args.seed
        if not args.use_priority:
            self.priority_prob_alpha = 0
        self.amp_type = args.amp_type
        self.use_priority = args.use_priority
        self.use_max_priority = args.use_max_priority if self.use_priority else False
        self.debug = args.debug
        self.device = args.device
        self.cpu_actor = args.cpu_actor
        self.gpu_actor = args.gpu_actor
        self.p_mcts_num = args.p_mcts_num
        self.use_root_value = args.use_root_value

        # Setup cluster specific config
        if args.cluster:
            self.ensemble_size = 5
            if args.case == 'deep_sea':
                self.ensemble_size = 10
                self.batch_size = 256
                # self.proj_hid = 1024
                # self.proj_out = 1024
                # self.pred_hid = 512
                # self.pred_out = 1024
            if args.case == 'atari':
                self.batch_size = 256
                # Base network arch.
                self.lstm_hidden_size = 512
                self.proj_hid = 1024
                self.proj_out = 1024
                self.pred_hid = 512
                self.pred_out = 1024

        # Setup deep_sea specific config
        if args.case == 'deep_sea':
            assert self.amp_type == 'none', "deep_sea is only implemented without torch_amp."
            # In deep sea w. MuExplore we want to update weights often in selfplay, to make the most of exploration
            # As a result, we compute checkpoint_interval as once every batched episode
            training_steps_per_episode_ratio = self.training_ratio * self.p_mcts_num * self.env_size
            self.checkpoint_interval = min(self.checkpoint_interval, 5)
            # We compute target_model_interval as once every M batched episodes.
            M = 1
            self.target_model_interval = min(self.target_model_interval, 10)
            # Reset ube every N batched episodes
            # N = 50
            # self.reset_ube_interval = min(self.reset_ube_interval, N * training_steps_per_episode_ratio)
            self.use_visitation_counter = args.visit_counter
            # only use visit_counter in planning when it's enabled
            self.plan_with_visitation_counter = args.p_w_vis_counter and self.use_visitation_counter
            self.sampling_times = args.sampling_times
            if self.plan_with_visitation_counter:
                self.plan_with_fake_visit_counter = args.plan_w_fake_visit_counter
                self.plan_with_state_visits = args.plan_w_state_visits
            self.architecture_type = 'fully_connected'     # Only fully_connected is implemented for deep_sea
            self.learned_model = not args.alpha_zero_planning   # AlphaZero is only implemented in deep_sea
            if not self.learned_model:
                self.do_consistency = False # Consistency loss only makes sense with a learned model.
            self.deepsea_randomize_actions = not args.det_deepsea_actions
            if args.learned_representation:
                self.identity_representation = False

            if args.representation_based_training:
                self.representation_based_training = args.representation_based_training

        # MuExplore:
        self.uncertainty_architecture_type = args.uncertainty_architecture_type
        self.use_uncertainty_architecture = args.uncertainty_architecture  # Is there an active unc. arch. (rnd / ens.)
        assert (args.mu_explore == (self.use_uncertainty_architecture or self.use_visitation_counter)) or (
            not args.mu_explore)  # MuExplore is only applicable with some uncertainty mechanism
        self.mu_explore = args.mu_explore and (self.use_uncertainty_architecture or self.use_visitation_counter)
        self.use_deep_exploration = self.mu_explore or 'ube' in self.uncertainty_architecture_type
        if args.beta is not None and args.beta >= 0:
            self.beta = args.beta * 1.0
        elif args.beta is not None and args.beta < 0:
            print(
                f"In parameter setup, received illegal beta value < 0. Setting the currently configured default beta "
                f"instead: beta = {self.beta}")
        self.standard_action_selection = not args.q_based_action_selection or not self.use_deep_exploration
        self.disable_policy_in_exploration = args.disable_policy_in_exploration
        self.root_exploration_fraction = args.exploration_fraction
        # Max value and policy targets can be used independently, but only with deep exploration
        self.use_max_value_targets = args.use_max_value_targets and self.use_deep_exploration
        self.use_max_policy_targets = args.use_max_policy_targets and self.use_deep_exploration

        if args.number_of_exploratory_envs is not None and self.use_deep_exploration:
            assert args.number_of_exploratory_envs <= self.p_mcts_num
            self.number_of_exploratory_envs = args.number_of_exploratory_envs
        else:
            # Number_of_exploratory_envs defaults to all envs - 1, if deep_exploration is used
            self.number_of_exploratory_envs = self.p_mcts_num - 1 if self.use_deep_exploration else 0

        # UBE parameters:
        if 'ube' in self.uncertainty_architecture_type:
            self.periodic_ube_weight_reset = args.periodic_ube_weight_reset
            self.count_based_ube = self.plan_with_visitation_counter

        if args.prior_scale is not None:
            self.prior_scale = args.prior_scale

        if not self.do_consistency:
            self.consistency_coeff = 0
            self.augmentation = None
            self.use_augmentation = False

        if not self.use_value_prefix:
            self.lstm_horizon_len = 1

        if not self.off_correction:
            self.auto_td_steps = self.training_steps
        else:
            self.auto_td_steps = self.auto_td_steps_ratio * self.training_steps

        assert 0 <= self.lr_warm_up <= 0.1
        assert 1 <= self.lstm_horizon_len <= self.num_unroll_steps
        assert self.start_transitions >= self.batch_size

        # augmentation
        if self.consistency_coeff > 0 and args.use_augmentation and 'deep_sea' not in args.case:
            self.use_augmentation = True
            self.augmentation = args.augmentation
        else:
            self.use_augmentation = False

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        localtime = time.asctime(time.localtime(time.time()))
        if self.mu_explore:
            seed_tag = 'mu_explore_seed={}'.format(self.seed)
        elif self.use_uncertainty_architecture:
            seed_tag = 'ensemble_baseline_seed={}'.format(self.seed)
        else:
            seed_tag = 'baseline_seed={}'.format(self.seed)

        # Setup path
        self.exp_path = os.path.join(args.result_dir, args.case, args.info, args.env, seed_tag, localtime)
        if args.cluster:
            cluster_results_path = os.path.join("/scratch/yanivoren/efficient_explore_results/", 'results')
            self.exp_path = os.path.join(cluster_results_path, args.case, args.info, args.env, seed_tag, localtime)

        self.model_path = os.path.join(self.exp_path, 'model.p')
        self.model_dir = os.path.join(self.exp_path, 'model')
        return self.exp_path
