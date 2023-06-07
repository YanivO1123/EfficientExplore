from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from plotters import plotting_utils as c_p
import os
import pathlib
import re


def load_results():
    """
        Loads the results for a specific experiment (for example, mountaincar exploratory counter uncertainty)
         with _ seeds (for example, 10) into one list, the way the smoother wants
    """
    root_path = "/home/yaniv/EfficientExplore/final_results/"
    model_types = ["learned_model", "true_model", "muzero_model"]
    agent_types = ["mctse", "random_baseline", "ube_baseline"]
    counts_name = "s_counts.npy"
    exploration_rate_names = ["states_visited_per_step.npy", "steps_for_states_visited.npy"]
    return_names = ["mean_test_results.npy", "training_steps.npy"]

    experiments = []

    for model_type in model_types:
        for agent_type in agent_types:
            path = root_path + model_type + "/" + agent_type + "/"
            dir_names = sorted([dir_name for dir_name in os.listdir(path)])
            for seed_dir in dir_names:
                # This assumes that all files are in this level of folders

                pass

    # Returns:
    # [returns_for_seed]
    return

    # Iterate over all seeds in the folder

    # load the results for all files of the right names in the folder

    # Start averaging and plotting

    # Make a list of the different seeds' folder names
    dir_names = sorted([dir_name for dir_name in os.listdir(path)])

    # Make a list of the separate files
    results_file_names = [
        [
            path + dir_name + "/" + filename
            for filename in sorted(
            os.listdir(path + dir_name)
        )
            if filename.endswith(".npy")
        ] for dir_name in dir_names
    ]

    experiment_results = []

    # For each seed
    for seed in results_file_names:
        # For each result-file
        for result_file in seed:
            if x_name[0] in result_file:
                new_file_name = os.path.dirname(result_file) + "/played_steps.npy"
                os.rename(result_file, new_file_name)
                xs = np.load(new_file_name)
            if x_name[1] in result_file or x_name[2] in result_file:
                xs = np.load(result_file, allow_pickle=True)
            elif y_name[0] in result_file or y_name[1] in result_file:
                ys = np.load(result_file)

        experiment_results.append([ys, xs])

    return experiment_results


# def plot_old_true_model_results():
#     # True Model
#     # Load all alpha_explore seeds that worked
#     # for alphaexplore
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/alphaexplore/deep_sea/15/"
#     indiv_paths = ["mu_explore_seed=1391267/Sun Apr 30 00:38:32 2023", "mu_explore_seed=7015372/Sun Apr 30 01:25:28 2023", "mu_explore_seed=4597644/Sun Apr 30 04:07:13 2023"]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["True_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # Load all alpha_ube seeds that worked
#     # for alpha_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/alpha_mcts_ube/deep_sea/15/"
#     indiv_paths = ["deep_exploration_no_muexp_seed=553140/Fri Apr 28 13:56:00 2023",
#                    "deep_exploration_no_muexp_seed=4874493/Fri Apr 28 13:56:01 2023",
#                    "deep_exploration_no_muexp_seed=9970288/Fri Apr 28 21:08:51 2023",
#                    "deep_exploration_no_muexp_seed=6118871/Sun Apr 30 04:07:14 2023",
#                    "deep_exploration_no_muexp_seed=556009/Sun Apr 30 04:39:36 2023",
#                    "deep_exploration_no_muexp_seed=3288310/Sun Apr 30 05:46:38 2023"]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["True_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, true model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, true model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_new_true_model_results():
#     # True Model
#     # Load all alpha_explore seeds that worked
#     # for alphaexplore
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/"
#     indiv_paths = [
#         "TrueMCTSE/deep_sea/15/mu_explore_seed=2488641/Mon May  1 12:16:56 2023",  # fine
#         "TrueMCTSE/deep_sea/15/mu_explore_seed=2896022/Mon May  1 12:20:45 2023",  # fine but didnt learn
#         "TrueMCTSE/deep_sea/15/mu_explore_seed=445311/Mon May  1 12:20:46 2023",  # fine but didnt learn
#         "TrueMCTSE/deep_sea/15/mu_explore_seed=5278963/Mon May  1 12:24:32 2023",  # fine but didnt learn
#         "TrueMCTSE/deep_sea/15/mu_explore_seed=9382637/Mon May  1 12:24:19 2023",  # fine but didnt learn
#         # "TrueMCTSE/deep_sea/15/mu_explore_seed=3443841/Mon May  1 13:07:02 2023"        # short
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["True_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # Load all alpha_ube seeds that worked
#     # for alpha_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/"
#     indiv_paths = ["TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=9854708//Mon May  1 13:15:55 2023",
#                    "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=3777342/Mon May  1 13:15:15 2023",
#                    "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=9527476/Mon May  1 15:19:10 2023",
#                    "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=1708939/Mon May  1 16:52:07 2023",
#                    "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=1684091/Mon May  1 16:53:20 2023"]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["True_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, true model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, true model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_learned_model_results():
#     # Learned Model
#     # For learned_MCTSE
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/muexplore/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = ["mu_explore_seed=9714621/Thu Apr 27 18:45:36 2023",
#                    "mu_explore_seed=3370714/Fri Apr 28 23:10:54 2023",
#                    "mu_explore_seed=8139193/Fri Apr 28 23:26:06 2023",
#                    "mu_explore_seed=941196/Fri Apr 28 10:19:57 2023",
#                    "mu_explore_seed=941196/Fri Apr 28 10:19:57 2023",
#                    "mu_explore_seed=9702782/Fri Apr 28 10:36:18 2023",
#                    "mu_explore_seed=1248903/Fri Apr 28 23:55:12 2023"]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["Learned_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/mcts_ube/deep_sea/15/"
#     indiv_paths = ["deep_exploration_no_muexp_seed=820422/Fri Apr 28 10:31:45 2023",
#                    "deep_exploration_no_muexp_seed=1702811/Fri Apr 28 10:33:19 2023",
#                    "deep_exploration_no_muexp_seed=4833302/Fri Apr 28 10:34:30 2023",
#                    "deep_exploration_no_muexp_seed=893414/Sat Apr 29 11:44:51 2023",
#                    "deep_exploration_no_muexp_seed=3853374/Sat Apr 29 11:49:58 2023",
#                    "deep_exploration_no_muexp_seed=4734238/Sat Apr 29 11:54:05 2023"]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["Learned_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, learned model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, learned model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_muzero_model():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = [
#         # "mu_explore_seed=5363468/Mon May  1 17:11:07 2023",
#                    "mu_explore_seed=699107/Mon May  1 18:06:25 2023"
#                    ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["Learned_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=5558597/Mon May  1 18:31:43 2023",
#         "deep_exploration_no_muexp_seed=9520949/Mon May  1 19:16:24 2023",
#         "deep_exploration_no_muexp_seed=9147919/Mon May  1 19:17:16 2023",
#         "deep_exploration_no_muexp_seed=8803842/Mon May  1 19:19:28 2023"
#                    ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["Learned_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_muzero_model_2():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = [
#         "mu_explore_seed=8236253/Mon May  8 13:18:17 2023",
#         "mu_explore_seed=2410451/Mon May  8 13:19:13 2023",
#         "mu_explore_seed=7404962/Mon May  8 13:39:16 2023"
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["Learned_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=7016396/Mon May  8 15:10:17 2023",
#         "deep_exploration_no_muexp_seed=4133396/Mon May  8 20:40:10 2023",
#         "deep_exploration_no_muexp_seed=6169874/Tue May  9 01:46:13 2023"
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["Learned_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_learned_mctse_2():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = [
#         "mu_explore_seed=8873744/Mon May  8 16:29:14 2023",
#         "mu_explore_seed=8266148/Mon May  8 16:29:47 2023",
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["Learned_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
#     indiv_paths = [
#         "",
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     # for path in indiv_paths:
#     #     local_path = main_path + path
#     #
#     #     counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#     #
#     #     visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#     #     steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#     #
#     #     average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#     #     training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#     #
#     # results_ube = [[returns, steps] for returns, steps in
#     #                zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     # experiment_names_ube = ["Learned_UBE"]
#     # results_ube_exploration = [[returns, steps] for returns, steps in
#     #                            zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore]#, results_ube]
#     experiment_names_return = [experiment_names_explore[0]]#, experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, learned model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration]#, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0]]#, experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, learned model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def plot_true_model_2():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
#     indiv_paths = [
#         "mu_explore_seed=5239126/Mon May  8 16:32:52 2023",
#         "mu_explore_seed=5270927/Mon May  8 16:33:49 2023"
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["True_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # Load all alpha_ube seeds that worked
#     # for alpha_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=7791068/Mon May  8 17:27:58 2023",
#         "deep_exploration_no_muexp_seed=1847642/Mon May  8 19:23:12 2023",
#         "deep_exploration_no_muexp_seed=7671529/Mon May  8 20:40:07 2023"
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["True_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, true model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, true model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
# def new_results_learned_model():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = [
#         "mu_explore_seed=1852012/Tue May  9 00:45:21 2023",
#         "mu_explore_seed=1891897/Tue May  9 06:47:20 2023",
#         "mu_explore_seed=1430628/Tue May  9 05:42:11 2023"
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["Learned_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=9977175/Tue May  9 11:18:16 2023",
#         "deep_exploration_no_muexp_seed=4715836/Tue May  9 11:19:31 2023",
#         "deep_exploration_no_muexp_seed=8350912/Tue May  9 11:39:49 2023"
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["Learned_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore , results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, learned model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, learned model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def new_results_true_model():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
#     indiv_paths = [
#         "mu_explore_seed=7911200/Tue May  9 00:53:37 2023",
#         "mu_explore_seed=8593574/Tue May  9 01:43:00 2023",
#         "mu_explore_seed=3218304/Tue May  9 05:56:42 2023",
#         "mu_explore_seed=1961392/Tue May  9 06:30:44 2023"
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["True_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # Load all alpha_ube seeds that worked
#     # for alpha_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=1755787/Tue May  9 11:31:48 2023",
#         "deep_exploration_no_muexp_seed=8804450/Tue May  9 11:40:48 2023",
#         "deep_exploration_no_muexp_seed=3038840/Tue May  9 14:23:08 2023"
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["True_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, true model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, true model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)
#
#
# def new_results_muzero_model():
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
#     indiv_paths = [
#         "mu_explore_seed=8518093/Tue May  9 14:46:22 2023",
#         "mu_explore_seed=562691/Tue May  9 14:47:33 2023",
#         "mu_explore_seed=6420563/Tue May  9 17:29:44 2023"
#     ]
#
#     counts_list_explore = []
#     visited_states_list_explore = []
#     steps_for_states_visited_list_explore = []  #
#     average_test_return_list_explore = []
#     training_steps_list_explore = []
#     labels_explore = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_explore.append(
#             np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))
#
#     results_explore = [[returns, steps] for returns, steps in
#                        zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
#     experiment_names_explore = ["MuZero_MCTSE"]
#     results_explore_exploration = [[returns, steps] for returns, steps in
#                                    zip(visited_states_list_explore, steps_for_states_visited_list_explore)]
#
#     # For learned_ube
#     main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
#     indiv_paths = [
#         "deep_exploration_no_muexp_seed=3242166/Tue May  9 15:25:01 2023",
#     ]
#
#     counts_list_ube = []
#     visited_states_list_ube = []
#     steps_for_states_visited_list_ube = []  #
#     average_test_return_list_ube = []
#     training_steps_list_ube = []
#     labels_ube = []
#
#     # load results for each seed
#     for path in indiv_paths:
#         local_path = main_path + path
#
#         counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
#
#         visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
#         steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
#
#         average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
#         training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
#
#     results_ube = [[returns, steps] for returns, steps in
#                    zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
#     experiment_names_ube = ["MuZero_UBE"]
#     results_ube_exploration = [[returns, steps] for returns, steps in
#                                zip(visited_states_list_ube, steps_for_states_visited_list_ube)]
#
#     # Plot against each other
#     results_return = [results_explore, results_ube]
#     experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Average return per step, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_return, experiment_names_return, sem=sem)
#     plt.legend(fontsize=6)
#
#     results_exploration = [results_explore_exploration, results_ube_exploration]
#     experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
#     sem = True
#     # plt.gcf()
#     plt.figure()
#     plt.title("Number of visited states, muzero model experiments", fontsize=8)
#     c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
#     plt.legend(fontsize=6)


def load_results_true_model(load_until_step=-1):
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
    indiv_paths = [
        # Older:
        # "mu_explore_seed=7911200/Tue May  9 00:53:37 2023",
        # "mu_explore_seed=8593574/Tue May  9 01:43:00 2023",
        # "mu_explore_seed=3218304/Tue May  9 05:56:42 2023",
        # "mu_explore_seed=1961392/Tue May  9 06:30:44 2023",
        # "final" With ube_td=1:
        "mu_explore_seed=807010/Fri May 12 11:26:34 2023",
        "mu_explore_seed=1402042/Fri May 12 21:48:52 2023",
        "mu_explore_seed=5713526/Fri May 12 21:47:57 2023",
        "mu_explore_seed=1258067/Fri May 12 22:29:08 2023",
        "mu_explore_seed=776223/Fri May 12 22:48:32 2023",
        # "mu_explore_seed=6721682/Mon May 15 00:51:15 2023",
        # "mu_explore_seed=4862938/Mon May 15 00:54:32 2023",
        # "mu_explore_seed=7630410/Mon May 15 01:06:32 2023"
    ]

    counts_list_explore, visited_states_list_explore, steps_for_states_visited_list_explore, average_test_return_list_explore, \
        training_steps_list_explore = load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS, RND"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
    indiv_paths = [
        # "deep_exploration_no_muexp_seed=1755787/Tue May  9 11:31:48 2023",
        # "deep_exploration_no_muexp_seed=8804450/Tue May  9 11:40:48 2023",
        # "deep_exploration_no_muexp_seed=3038840/Tue May  9 14:23:08 2023"
        # "final" With ube_td=1:
        "deep_exploration_no_muexp_seed=680609/Fri May 12 11:30:23 2023",
        "deep_exploration_no_muexp_seed=7409571/Fri May 12 11:30:22 2023",
        "deep_exploration_no_muexp_seed=183369/Fri May 12 11:30:28 2023",
        "deep_exploration_no_muexp_seed=5507605/Fri May 12 11:39:51 2023",
        "deep_exploration_no_muexp_seed=4944411/Fri May 12 12:56:03 2023",
        # "deep_exploration_no_muexp_seed=1956939/Mon May 15 05:08:16 2023",
        # "deep_exploration_no_muexp_seed=1213761/Mon May 15 05:18:47 2023"
    ]

    counts_list_ube, visited_states_list_ube, steps_for_states_visited_list_ube, average_test_return_list_ube, \
        training_steps_list_ube = load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE, RND"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    titles = ["Average return per step, true model experiments", "Number of visited states, true model experiments"]

    # For learned_vanilla
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueVanilla/deep_sea/15/"
    indiv_paths = [
        # "baseline_seed=2591953/Wed May 10 11:04:12 2023",
        # "baseline_seed=8363129/Wed May 10 11:10:56 2023"
        "baseline_seed=4029548/Fri May 12 13:05:55 2023",
        "baseline_seed=5086479/Fri May 12 14:49:21 2023",
        "baseline_seed=3257578/Fri May 12 17:13:51 2023",
        "baseline_seed=3589627/Fri May 12 17:29:09 2023",
        "baseline_seed=835636/Fri May 12 17:36:40 2023"
    ]

    counts_list_vanilla, visited_states_list_vanilla, steps_for_states_visited_list_vanilla, \
        average_test_return_list_vanilla, training_steps_list_vanilla = \
        load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_vanilla = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_vanilla, training_steps_list_vanilla)]  # Of form [[ys, xs], ...]
    experiment_names_vanilla = ["Uninformed"]
    results_vanilla_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_vanilla, steps_for_states_visited_list_vanilla)]

    # Plot against each other
    results_return = [results_explore, results_ube, results_vanilla]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]
    results_exploration = [results_explore_exploration, results_ube_exploration, results_vanilla_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration


def load_results_learned_model(load_until_step=-1):
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        # Older:
        # "mu_explore_seed=1852012/Tue May  9 00:45:21 2023",
        # "mu_explore_seed=1891897/Tue May  9 06:47:20 2023",
        # "mu_explore_seed=1430628/Tue May  9 05:42:11 2023"
        # With UBE_td=1:
        "mu_explore_seed=6750254/Fri May 12 20:09:58 2023",
        "mu_explore_seed=1617166/Fri May 12 20:11:10 2023",
        "mu_explore_seed=4071272/Fri May 12 21:38:25 2023",
        "mu_explore_seed=7935415/Fri May 12 21:48:52 2023",
        "mu_explore_seed=8178700/Sat May 13 02:36:38 2023",
        "mu_explore_seed=8714556/Sun May 14 19:22:03 2023",
        "mu_explore_seed=2323238/Sun May 14 19:26:26 2023",
        "mu_explore_seed=5237265/Sun May 14 19:28:40 2023",
        "mu_explore_seed=6508841/Sun May 14 19:31:18 2023",
        "mu_explore_seed=4720012/Sun May 14 19:37:50 2023"
    ]

    counts_list_explore, visited_states_list_explore, steps_for_states_visited_list_explore, average_test_return_list_explore, \
        training_steps_list_explore = load_results_individual_model(main_path, indiv_paths,
                                                                    load_until_step=load_until_step)

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS, RND"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
    indiv_paths = [
        # Older:
        # "deep_exploration_no_muexp_seed=9977175/Tue May  9 11:18:16 2023",
        # "deep_exploration_no_muexp_seed=4715836/Tue May  9 11:19:31 2023",
        # "deep_exploration_no_muexp_seed=8350912/Tue May  9 11:39:49 2023"
        # With ube_td=1:
        "deep_exploration_no_muexp_seed=7585832/Fri May 12 11:26:34 2023",
        "deep_exploration_no_muexp_seed=7000014/Fri May 12 11:26:30 2023",
        "deep_exploration_no_muexp_seed=7944487/Fri May 12 11:26:33 2023",
        "deep_exploration_no_muexp_seed=7961617/Fri May 12 11:26:31 2023",
        "deep_exploration_no_muexp_seed=9492941/Fri May 12 11:30:25 2023",
        "deep_exploration_no_muexp_seed=4332689/Sun May 14 19:48:20 2023",
        "deep_exploration_no_muexp_seed=4332689/Sun May 14 19:48:20 2023",
        "deep_exploration_no_muexp_seed=4332689/Sun May 14 19:48:20 2023",
        "deep_exploration_no_muexp_seed=1692791/Sun May 14 21:15:41 2023",
        "deep_exploration_no_muexp_seed=1697686/Mon May 15 00:17:01 2023"
    ]

    counts_list_ube, visited_states_list_ube, steps_for_states_visited_list_ube, average_test_return_list_ube, \
        training_steps_list_ube = load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE, RND"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    titles = ["Average return per step, anchored model experiments",
                           "Number of visited states, anchored model experiments"]

    # For learned_vanilla
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedVanilla/deep_sea/15/"
    indiv_paths = [
        "baseline_seed=9920413/Fri May 12 12:55:27 2023",
        "baseline_seed=6051942/Fri May 12 12:56:02 2023",
        "baseline_seed=1385612/Fri May 12 12:56:04 2023",
        "baseline_seed=5908967/Fri May 12 12:55:19 2023",
        "baseline_seed=4025319/Fri May 12 13:04:11 2023"
    ]

    counts_list_vanilla, visited_states_list_vanilla, steps_for_states_visited_list_vanilla, \
        average_test_return_list_vanilla, training_steps_list_vanilla = \
        load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_vanilla = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_vanilla, training_steps_list_vanilla)]  # Of form [[ys, xs], ...]
    experiment_names_vanilla = ["Uninformed"]
    results_vanilla_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_vanilla, steps_for_states_visited_list_vanilla)]

    # Plot against each other
    results_return = [results_explore, results_ube, results_vanilla]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]

    results_exploration = [results_explore_exploration, results_ube_exploration, results_vanilla_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration


def load_results_muzero_model(load_until_step=-1):
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        # Older:
        # "mu_explore_seed=8518093/Tue May  9 14:46:22 2023",
        # "mu_explore_seed=562691/Tue May  9 14:47:33 2023",
        # "mu_explore_seed=6420563/Tue May  9 17:29:44 2023"
        # UBE_td=5:
        # "mu_explore_seed=8167756/Thu May 11 11:03:27 2023",
        # "mu_explore_seed=5695122/Thu May 11 11:07:31 2023",
        # "mu_explore_seed=1265005/Thu May 11 11:10:53 2023",
        # "mu_explore_seed=8929823/Thu May 11 11:21:31 2023",
        # "mu_explore_seed=264459/Thu May 11 11:24:10 2023"
        # UBE_td=1, all those that lasted around 35k:
        "mu_explore_seed=479364/Sat May 13 01:19:21 2023", # 34
        "mu_explore_seed=4944454/Sat May 13 01:48:24 2023", # 34
        "mu_explore_seed=4756/Sat May 13 17:56:45 2023",    # 34
        "mu_explore_seed=1926033/Sat May 13 18:12:32 2023", # 34
        "mu_explore_seed=107875/Sat May 13 18:32:52 2023",  # 35
        # "mu_explore_seed=2204985/Sat May 13 23:48:50 2023", # 35
        # "mu_explore_seed=9122315/Sun May 14 05:03:42 2023",     # 35
        # "mu_explore_seed=4961671/Sun May 14 05:40:44 2023",  # 34.5

        # "mu_explore_seed=420478/Sun May 14 10:40:25 2023",  # 40k
        # "mu_explore_seed=3719327/Sun May 14 10:40:25 2023", # 41k
        # "mu_explore_seed=4903080/Sun May 14 10:38:57 2023"    # finished
    ]

    counts_list_explore, visited_states_list_explore, steps_for_states_visited_list_explore, average_test_return_list_explore, \
        training_steps_list_explore = load_results_individual_model(main_path, indiv_paths,
                                                                    load_until_step=load_until_step)

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS, counts"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
    indiv_paths = [
        # Older:
        # "deep_exploration_no_muexp_seed=3242166/Tue May  9 15:25:01 2023",
        # With ube_td=5:
        # "deep_exploration_no_muexp_seed=6747263/Thu May 11 14:22:37 2023",
        # "deep_exploration_no_muexp_seed=2694818/Thu May 11 14:32:58 2023",
        # "deep_exploration_no_muexp_seed=178134/Thu May 11 14:50:50 2023",
        # "deep_exploration_no_muexp_seed=7259110/Thu May 11 15:29:52 2023",
        # "deep_exploration_no_muexp_seed=9265132/Thu May 11 15:29:52 2023",
        # With ube_td=1:
        "deep_exploration_no_muexp_seed=8900607/Fri May 12 23:03:15 2023",
        "deep_exploration_no_muexp_seed=8926050/Fri May 12 23:14:26 2023",
        "deep_exploration_no_muexp_seed=7833116/Fri May 12 23:26:36 2023",
        "deep_exploration_no_muexp_seed=5196822/Sat May 13 00:25:38 2023",
        "deep_exploration_no_muexp_seed=7202327/Sat May 13 00:28:39 2023",
    ]

    counts_list_ube, visited_states_list_ube, steps_for_states_visited_list_ube, average_test_return_list_ube, \
        training_steps_list_ube = load_results_individual_model(main_path, indiv_paths,
                                                                load_until_step=load_until_step)

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE, counts"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    titles = ["Average return per step, abstracted model experiments", "Number of visited states, abstracted model experiments"]

    # For learned_vanilla
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuZero/deep_sea/15/"
    indiv_paths = [
        "baseline_seed=2286067/Thu May 11 15:58:46 2023",
        "baseline_seed=1272961/Thu May 11 17:24:43 2023",
        "baseline_seed=3009178/Fri May 12 18:26:24 2023",
        "baseline_seed=2608286/Fri May 12 19:51:52 2023",
        "baseline_seed=6935859/Sat May 13 19:47:11 2023"
    ]

    counts_list_vanilla, visited_states_list_vanilla, steps_for_states_visited_list_vanilla, \
        average_test_return_list_vanilla, training_steps_list_vanilla = \
        load_results_individual_model(main_path, indiv_paths, load_until_step=load_until_step)

    results_vanilla = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_vanilla, training_steps_list_vanilla)]  # Of form [[ys, xs], ...]
    experiment_names_vanilla = ["Uninformed"]
    results_vanilla_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_vanilla, steps_for_states_visited_list_vanilla)]

    # Plot against each other
    results_return = [results_explore, results_ube, results_vanilla]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]
    results_exploration = [results_explore_exploration, results_ube_exploration, results_vanilla_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0], experiment_names_vanilla[0]]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration


def load_results_individual_model(main_path, indiv_paths, load_until_step=-1):
    counts_list = []
    visited_states_list = []
    steps_for_states_visited_list = []  #
    average_test_return_list = []
    training_steps_list = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts = np.load(local_path + "/s_counts.npy")
        visited_states = np.load(local_path + "/states_visited_per_step.npy")
        steps_for_states_visited = np.load(local_path + "/steps_for_states_visited.npy")
        average_test_return = np.load(local_path + "/mean_test_results.npy")
        training_steps = np.load(local_path + "/training_steps.npy")

        if 'Mu' in local_path:
            # Find min load_until_step index
            load_until_step = 35000
            training_steps_index = np.searchsorted(training_steps, load_until_step, side='right')
            steps_for_states_visited_index = np.searchsorted(steps_for_states_visited, load_until_step, side='right')

            # remove all entries after this training step
            visited_states = visited_states[:steps_for_states_visited_index]
            steps_for_states_visited = steps_for_states_visited[
                                       :steps_for_states_visited_index]  # / 40  # this will translate to episodes
            average_test_return = average_test_return[:training_steps_index]
            training_steps = training_steps[:training_steps_index]  # / 40 # this will translate to episodes
        elif load_until_step > 0:
            # Find min load_until_step index
            training_steps_index = np.searchsorted(training_steps, load_until_step, side='right')
            steps_for_states_visited_index = np.searchsorted(steps_for_states_visited, load_until_step, side='right')

            # remove all entries after this training step
            visited_states = visited_states[:steps_for_states_visited_index]
            steps_for_states_visited = steps_for_states_visited[:steps_for_states_visited_index]# / 40  # this will translate to episodes
            average_test_return = average_test_return[:training_steps_index]
            training_steps = training_steps[:training_steps_index]# / 40 # this will translate to episodes


        counts_list.append(counts)
        visited_states_list.append(visited_states)
        steps_for_states_visited_list.append(steps_for_states_visited)
        average_test_return_list.append(average_test_return)
        training_steps_list.append(training_steps)

    return counts_list, visited_states_list, steps_for_states_visited_list, average_test_return_list, training_steps_list


def load_final_results(load_until_step=-1):
    subplot_titles = []  # List of shape [plots, 2]
    final_results_return = []  # List of shape [plots, ...]
    final_experiment_names_returns = []  # List of shape [plots, ...]
    final_results_exploration = []  # List of shape [plots, ...]
    final_experiment_names_exploration = []  # List of shape [plots, ...]

    ### True Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_true_model(load_until_step)
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    ### MuZero Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_muzero_model(load_until_step)
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    ### Learned Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_learned_model(
        load_until_step)
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    return subplot_titles, final_results_return, final_experiment_names_returns, final_results_exploration, final_experiment_names_exploration


def plot_joined():
    subplot_titles, results_return, experiment_names_returns, results_exploration, experiment_names_exploration = load_final_results(load_until_step=45000)
    plot_until = 35000
    sem = True
    plt.figure(0)
    index = 0
    num_plots = 6

    for i in range(int(num_plots / 2)):
        subplot_shape = 3 * 100 + 2 * 10 + index + 1
        plt.subplot(subplot_shape)
        # plt.title(subplot_titles[i][0], fontsize=8)
        c_p.plot_log(results_return[i], experiment_names_returns[i], sem=sem, bins=200)
        # plt.legend(fontsize=6)
        plt.subplot(subplot_shape + 1)
        # plt.title(subplot_titles[i][1], fontsize=8)
        c_p.plot_exploration(results_exploration[i], experiment_names_exploration[i], sem=sem, bins=200, plot_zero=False)
        if i < 2:
            x = np.linspace(0, 45000, num=200, endpoint=True)
        else:
            x = np.linspace(0, 34000, num=200, endpoint=True)
        # Define the y-value to plot a dotted line at
        y_value = 820
        # Create a list of y-values with the same length as x, all equal to y_value
        y = [y_value] * len(x)
        plt.plot(x, y, linestyle=':', color='black', label='maximum')
        plt.legend(fontsize=6)#, loc='lower right')
        index += 2

    plt.tight_layout()


def plot_joined_subfigs():
    subplot_titles, results_return, experiment_names_returns, results_exploration, experiment_names_exploration = load_final_results(
        load_until_step=45000)
    sem = True

    fig = plt.figure(constrained_layout=True)

    suptitles = ['True Model', 'Abstracted Model', 'Anchored Model']

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(suptitles[row], fontsize=8)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2)
        for col, ax in enumerate(axs):
            if col == 0:
                c_p.plot_log_ax(results_return[row], experiment_names_returns[row], ax, sem=sem, bins=50)
            else:
                c_p.plot_exploration_ax(results_exploration[row], experiment_names_exploration[row], ax,
                                    sem=sem, bins=200,
                                    plot_zero=False)
                if row == 1:
                    x = np.linspace(0, 35000, num=200, endpoint=True)
                else:
                    x = np.linspace(0, 45000, num=200, endpoint=True)

                # Define the y-value to plot a dotted line at
                y_value = 820
                # Create a list of y-values with the same length as x, all equal to y_value
                y = [y_value] * len(x)
                ax.plot(x, y, linestyle=':', color='black', label='maximum')
                ax.legend(fontsize=6)  # , loc='lower right')
    # plt.tight_layout()


def plot_joined_for_dagstuhl():
    subplot_titles, results_return, experiment_names_returns, results_exploration, experiment_names_exploration = load_final_results(
        load_until_step=45000)
    sem = True

    fig = plt.figure(constrained_layout=True)

    suptitles = ['True Model', 'Abstracted Model', 'Anchored Model']

    # create 2 x 2x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(suptitles[row], fontsize=8)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2)
        for col, ax in enumerate(axs):
            if col == 0:
                c_p.plot_log_ax(results_return[row], experiment_names_returns[row], ax, sem=sem, bins=50)
            else:
                c_p.plot_exploration_ax(results_exploration[row], experiment_names_exploration[row], ax,
                                        sem=sem, bins=200,
                                        plot_zero=False)
                if row == 1:
                    x = np.linspace(0, 35000, num=200, endpoint=True)
                else:
                    x = np.linspace(0, 45000, num=200, endpoint=True)

                # Define the y-value to plot a dotted line at
                y_value = 820
                # Create a list of y-values with the same length as x, all equal to y_value
                y = [y_value] * len(x)
                ax.plot(x, y, linestyle=':', color='black', label='maximum')
                ax.legend(fontsize=6)  # , loc='lower right')

    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(suptitles[row + 1], fontsize=8)

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2)
        for col, ax in enumerate(axs):
            if col == 0:
                c_p.plot_log_ax(results_return[row + 1], experiment_names_returns[row + 1], ax, sem=sem, bins=50)
            else:
                c_p.plot_exploration_ax(results_exploration[row + 1], experiment_names_exploration[row + 1], ax,
                                        sem=sem, bins=200,
                                        plot_zero=False)
                if row == 0:
                    x = np.linspace(0, 35000, num=200, endpoint=True)
                else:
                    x = np.linspace(0, 45000, num=200, endpoint=True)

                # Define the y-value to plot a dotted line at
                y_value = 820
                # Create a list of y-values with the same length as x, all equal to y_value
                y = [y_value] * len(x)
                ax.plot(x, y, linestyle=':', color='black', label='maximum')
                ax.legend(fontsize=6)  # , loc='lower right')


# Keeps track of recent local experiment
# path = "/home/yaniv/EfficientExplore/results/deep_sea/MuMCTSE/deep_sea/15/mu_explore_seed=7038764/Tue May  9 23:14:33 2023/s_counts.npy"  # _at_step_5000 _at_step_10000
# counts = np.load(path)
# c_p.plot_heat_maps(counts, sa_counts=None, count_cap=20)
# plt.show()

# Take 3, joined
# plot_joined()
# plot_joined_subfigs()
plot_joined_for_dagstuhl()

plt.show()
exit()