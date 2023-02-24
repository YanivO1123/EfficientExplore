import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari', 'deep_sea'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--amp_type', required=True, choices=['torch_amp', 'none'],
                        help='choose automated mixed precision type')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--save_video', action='store_true', default=False, help='save video in test.')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int, default=14, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int, default=20, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=4, help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=None, help='seed (defaults to: choose a seed at random)')
    parser.add_argument('--num_gpus', type=int, default=4, help='gpus available')
    parser.add_argument('--num_cpus', type=int, default=80, help='cpus available')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true', default=False,
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true', default=False,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_max_priority', action='store_true', default=False, help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none', help='debug string')
    parser.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
    parser.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int, default=None, help='object store memory') # original default=150 * 1024 * 1024 * 1024
    parser.add_argument('--cluster', action='store_true', default=False,
                        help="Is used for switching experiment configurations between debugging (local, False) and final "
                             "(computation cluster, True)")
    parser.add_argument('--beta', type=float, default=None, help='Exploration / exploitation parameter, takes float >= 0')
    parser.add_argument('--mu_explore', action='store_true', default=False,
                        help="Use MuExplore (exploratory MCTS), or not.")
    parser.add_argument('--uncertainty_architecture', action='store_true', default=False,
                        help="Use uncertainty_architecture, or not.")
    parser.add_argument('--disable_policy_in_exploration', action='store_true', default=False,
                        help="If using MuExplore, disable policy-prior node scores in MCTS search in exploration episodes. "
                             "If false, can be too policy-biased and not provide effective exploration.")
    parser.add_argument('--exploration_fraction', type=float, default=0.25,
                        help='noise magnitude to add to nodes in MCTS. Defaults to 0.25 used by EffZero')
    parser.add_argument('--visit_counter', action='store_true', default=False,
                        help="If the env. is deep sea, use the visit counter for uncertainty estimation. "
                             "Otherwise, does nothing.")
    parser.add_argument('--p_w_vis_counter', action='store_true', default=False,
                        help="If the env. is deep sea, use the visit counter in MCTS planning with muexplore. "
                             "Otherwise, does nothing.")
    parser.add_argument('--use_max_value_targets', action='store_true', default=False,
                        help="Use max targets in exploratory episodes. Only applicable with MuExplore. If not specified,"
                             "uses use_max_value_targets from the config (which defaults to false unless set otherwise)")
    parser.add_argument('--use_max_policy_targets', action='store_true', default=False,
                        help="Use max policy targets from max value targets in exploratory episodes. Only applicable with MuExplore AND use_max_value_targets. If not specified,"
                             "uses use_max_value_targets from the config (which defaults to false unless set otherwise)")
    parser.add_argument('--plan_w_fake_visit_counter', action='store_true', default=False,
                        help="For debugging. If true, unc. associated with rewarding state is always maximized")
    parser.add_argument('--plan_w_state_visits', action='store_true', default=False,
                        help="If true uses state visits. Otherwise, will use state-action visits."
                             "Only relevant if uses p_w_vis_counter, ")
    parser.add_argument('--uncertainty_architecture_type', required=False, choices=['ensemble', 'rnd', 'rnd_ube', 'ensemble_ube'], default='ensemble',
                        help="Decides the type of uncertainty to be used.")
    parser.add_argument('--number_of_exploratory_envs', type=int, default=None, help='If MuExplore, number of environments <= p_mcts_num that'
                                                                                     'are exploratory')
    parser.add_argument('--det_deepsea_actions', action='store_true', default=False,
                        help="If true, use determinstic deep sea actions ")
    parser.add_argument('--architecture_type', required=False,
                        choices=['resnet', 'fully_connected'], default='resnet',
                        help="Resnet (original), and fully-connected (custom and only applicable to Deep Sea)")
    parser.add_argument('--sampling_times', type=int, default=0, help='If MuExplore and visitation counter, '
                                                                      'whether to use the sampled value '
                                                                      'propagation or not. Defaults to not')
    parser.add_argument('--alpha_zero_planning', action='store_true', default=False,
                        help="Only applied to deep_sea. If true, the dynamics model is not learned, but the true model "
                             "is used instead. Does not modify reward or value learning.")



    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    if args.opr == 'train':
        if args.object_store_memory is not None:
            ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus,
                object_store_memory=args.object_store_memory) # , RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1)
        else:
            ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus)
    else:
        ray.init()

    # seeding random iterators
    if args.seed is None:
        low = 0
        high = 10000000
        seed = np.random.randint(low, high=high)
        args.seed = seed
        print(f"setting up randomly-chosen seed. seed = {args.seed}", flush=True)

    set_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        from config.atari import game_config
    elif args.case == 'deep_sea':
        from config.deepsea import game_config
    else:
        raise Exception('Invalid --case option')

    # set config as per arguments
    exp_path = game_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger('train').info('Path: {}'.format(exp_path))
    logging.getLogger('train').info('Param: {}'.format(game_config.get_hparams()))

    device = game_config.device
    try:
        if args.opr == 'train':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.load_model and os.path.exists(args.model_path):
                model_path = args.model_path
            else:
                model_path = None
            model, weights = train(game_config, summary_writer, model_path)
            model.set_weights(weights)
            total_steps = game_config.training_steps + game_config.last_steps
            test_score, _, test_path = test(game_config, model.to(device), total_steps, game_config.test_episodes, device, render=False, save_video=args.save_video, final_test=True, use_pb=True)
            mean_score = test_score.mean()
            std_score = test_score.std()

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
            }
            for key, val in test_log.items():
                summary_writer.add_scalar('train/{}'.format(key), np.mean(val), total_steps)

            test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                       ''.format(total_steps, game_config.env_name, mean_score, test_score.max(), test_score.min(), std_score)
            logging.getLogger('train_test').info(test_msg)
            if args.save_video:
                logging.getLogger('train_test').info('Saving video in path: {}'.format(test_path))
        elif args.opr == 'test':
            assert args.load_model
            if args.model_path is None:
                model_path = game_config.model_path
            else:
                model_path = args.model_path
            assert os.path.exists(model_path), 'model not found at {}'.format(model_path)

            model = game_config.get_uniform_network().to(device)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            test_score, _, test_path = test(game_config, model, 0, args.test_episodes, device=device, render=args.render, save_video=args.save_video, final_test=True, use_pb=True)
            mean_score = test_score.mean()
            std_score = test_score.std()
            logging.getLogger('test').info('Test Mean Score: {} (max: {}, min: {})'.format(mean_score, test_score.max(), test_score.min()))
            logging.getLogger('test').info('Test Std Score: {}'.format(std_score))
            if args.save_video:
                logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
        else:
            raise Exception('Please select a valid operation(--opr) to be performed')
        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
