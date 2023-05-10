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


def plot_old_true_model_results():
    # True Model
    # Load all alpha_explore seeds that worked
    # for alphaexplore
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/alphaexplore/deep_sea/15/"
    indiv_paths = ["mu_explore_seed=1391267/Sun Apr 30 00:38:32 2023", "mu_explore_seed=7015372/Sun Apr 30 01:25:28 2023", "mu_explore_seed=4597644/Sun Apr 30 04:07:13 2023"]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["True_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/alpha_mcts_ube/deep_sea/15/"
    indiv_paths = ["deep_exploration_no_muexp_seed=553140/Fri Apr 28 13:56:00 2023",
                   "deep_exploration_no_muexp_seed=4874493/Fri Apr 28 13:56:01 2023",
                   "deep_exploration_no_muexp_seed=9970288/Fri Apr 28 21:08:51 2023",
                   "deep_exploration_no_muexp_seed=6118871/Sun Apr 30 04:07:14 2023",
                   "deep_exploration_no_muexp_seed=556009/Sun Apr 30 04:39:36 2023",
                   "deep_exploration_no_muexp_seed=3288310/Sun Apr 30 05:46:38 2023"]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["True_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, true model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, true model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_new_true_model_results():
    # True Model
    # Load all alpha_explore seeds that worked
    # for alphaexplore
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/"
    indiv_paths = [
        "TrueMCTSE/deep_sea/15/mu_explore_seed=2488641/Mon May  1 12:16:56 2023",  # fine
        "TrueMCTSE/deep_sea/15/mu_explore_seed=2896022/Mon May  1 12:20:45 2023",  # fine but didnt learn
        "TrueMCTSE/deep_sea/15/mu_explore_seed=445311/Mon May  1 12:20:46 2023",  # fine but didnt learn
        "TrueMCTSE/deep_sea/15/mu_explore_seed=5278963/Mon May  1 12:24:32 2023",  # fine but didnt learn
        "TrueMCTSE/deep_sea/15/mu_explore_seed=9382637/Mon May  1 12:24:19 2023",  # fine but didnt learn
        # "TrueMCTSE/deep_sea/15/mu_explore_seed=3443841/Mon May  1 13:07:02 2023"        # short
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["True_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/"
    indiv_paths = ["TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=9854708//Mon May  1 13:15:55 2023",
                   "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=3777342/Mon May  1 13:15:15 2023",
                   "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=9527476/Mon May  1 15:19:10 2023",
                   "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=1708939/Mon May  1 16:52:07 2023",
                   "TrueUBE/deep_sea/15/deep_exploration_no_muexp_seed=1684091/Mon May  1 16:53:20 2023"]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["True_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, true model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, true model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_learned_model_results():
    # Learned Model
    # For learned_MCTSE
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/muexplore/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = ["mu_explore_seed=9714621/Thu Apr 27 18:45:36 2023",
                   "mu_explore_seed=3370714/Fri Apr 28 23:10:54 2023",
                   "mu_explore_seed=8139193/Fri Apr 28 23:26:06 2023",
                   "mu_explore_seed=941196/Fri Apr 28 10:19:57 2023",
                   "mu_explore_seed=941196/Fri Apr 28 10:19:57 2023",
                   "mu_explore_seed=9702782/Fri Apr 28 10:36:18 2023",
                   "mu_explore_seed=1248903/Fri Apr 28 23:55:12 2023"]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["Learned_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/mcts_ube/deep_sea/15/"
    indiv_paths = ["deep_exploration_no_muexp_seed=820422/Fri Apr 28 10:31:45 2023",
                   "deep_exploration_no_muexp_seed=1702811/Fri Apr 28 10:33:19 2023",
                   "deep_exploration_no_muexp_seed=4833302/Fri Apr 28 10:34:30 2023",
                   "deep_exploration_no_muexp_seed=893414/Sat Apr 29 11:44:51 2023",
                   "deep_exploration_no_muexp_seed=3853374/Sat Apr 29 11:49:58 2023",
                   "deep_exploration_no_muexp_seed=4734238/Sat Apr 29 11:54:05 2023"]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Learned_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_muzero_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        # "mu_explore_seed=5363468/Mon May  1 17:11:07 2023",
                   "mu_explore_seed=699107/Mon May  1 18:06:25 2023"
                   ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["Learned_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=5558597/Mon May  1 18:31:43 2023",
        "deep_exploration_no_muexp_seed=9520949/Mon May  1 19:16:24 2023",
        "deep_exploration_no_muexp_seed=9147919/Mon May  1 19:17:16 2023",
        "deep_exploration_no_muexp_seed=8803842/Mon May  1 19:19:28 2023"
                   ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Learned_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_muzero_model_2():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=8236253/Mon May  8 13:18:17 2023",
        "mu_explore_seed=2410451/Mon May  8 13:19:13 2023",
        "mu_explore_seed=7404962/Mon May  8 13:39:16 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["Learned_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=7016396/Mon May  8 15:10:17 2023",
        "deep_exploration_no_muexp_seed=4133396/Mon May  8 20:40:10 2023",
        "deep_exploration_no_muexp_seed=6169874/Tue May  9 01:46:13 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Learned_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_learned_mctse_2():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=8873744/Mon May  8 16:29:14 2023",
        "mu_explore_seed=8266148/Mon May  8 16:29:47 2023",
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["Learned_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
    indiv_paths = [
        "",
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    # for path in indiv_paths:
    #     local_path = main_path + path
    #
    #     counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000
    #
    #     visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
    #     steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))
    #
    #     average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
    #     training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))
    #
    # results_ube = [[returns, steps] for returns, steps in
    #                zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    # experiment_names_ube = ["Learned_UBE"]
    # results_ube_exploration = [[returns, steps] for returns, steps in
    #                            zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore]#, results_ube]
    experiment_names_return = [experiment_names_explore[0]]#, experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration]#, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0]]#, experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def plot_true_model_2():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
    indiv_paths = [
        "mu_explore_seed=5239126/Mon May  8 16:32:52 2023",
        "mu_explore_seed=5270927/Mon May  8 16:33:49 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["True_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=7791068/Mon May  8 17:27:58 2023",
        "deep_exploration_no_muexp_seed=1847642/Mon May  8 19:23:12 2023",
        "deep_exploration_no_muexp_seed=7671529/Mon May  8 20:40:07 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["True_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, true model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, true model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)

def new_results_learned_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=1852012/Tue May  9 00:45:21 2023",
        "mu_explore_seed=1891897/Tue May  9 06:47:20 2023",
        "mu_explore_seed=1430628/Tue May  9 05:42:11 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["Learned_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=9977175/Tue May  9 11:18:16 2023",
        "deep_exploration_no_muexp_seed=4715836/Tue May  9 11:19:31 2023",
        "deep_exploration_no_muexp_seed=8350912/Tue May  9 11:39:49 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Learned_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore , results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, learned model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def new_results_true_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
    indiv_paths = [
        "mu_explore_seed=7911200/Tue May  9 00:53:37 2023",
        "mu_explore_seed=8593574/Tue May  9 01:43:00 2023",
        "mu_explore_seed=3218304/Tue May  9 05:56:42 2023",
        "mu_explore_seed=1961392/Tue May  9 06:30:44 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["True_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=1755787/Tue May  9 11:31:48 2023",
        "deep_exploration_no_muexp_seed=8804450/Tue May  9 11:40:48 2023",
        "deep_exploration_no_muexp_seed=3038840/Tue May  9 14:23:08 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["True_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, true model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, true model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def new_results_muzero_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=8518093/Tue May  9 14:46:22 2023",
        "mu_explore_seed=562691/Tue May  9 14:47:33 2023",
        "mu_explore_seed=6420563/Tue May  9 17:29:44 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["MuZero_MCTSE"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=3242166/Tue May  9 15:25:01 2023",
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["MuZero_UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average return per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_return, experiment_names_return, sem=sem)
    plt.legend(fontsize=6)

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    sem = True
    # plt.gcf()
    plt.figure()
    plt.title("Average states visited so far per step, muzero model experiments", fontsize=8)
    c_p.plot_log(results_exploration, experiment_exploration, sem=sem)
    plt.legend(fontsize=6)


def load_results_learned_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=1852012/Tue May  9 00:45:21 2023",
        "mu_explore_seed=1891897/Tue May  9 06:47:20 2023",
        "mu_explore_seed=1430628/Tue May  9 05:42:11 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/LearnedUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=9977175/Tue May  9 11:18:16 2023",
        "deep_exploration_no_muexp_seed=4715836/Tue May  9 11:19:31 2023",
        "deep_exploration_no_muexp_seed=8350912/Tue May  9 11:39:49 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]

    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0]]

    titles = ["Average return per step, anchored model experiments",
                           "Average states visited so far per step, anchored model experiments"]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration


def load_results_true_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueMCTSE/deep_sea/15/"
    indiv_paths = [
        "mu_explore_seed=7911200/Tue May  9 00:53:37 2023",
        "mu_explore_seed=8593574/Tue May  9 01:43:00 2023",
        "mu_explore_seed=3218304/Tue May  9 05:56:42 2023",
        "mu_explore_seed=1961392/Tue May  9 06:30:44 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # Load all alpha_ube seeds that worked
    # for alpha_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/TrueUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=1755787/Tue May  9 11:31:48 2023",
        "deep_exploration_no_muexp_seed=8804450/Tue May  9 11:40:48 2023",
        "deep_exploration_no_muexp_seed=3038840/Tue May  9 14:23:08 2023"
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0]]

    titles = ["Average return per step, true model experiments", "Average states visited so far per step, true model experiments"]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration


def load_results_muzero_model():
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuMCTSE/deep_sea/15/"  # for MCTSE_learned
    indiv_paths = [
        "mu_explore_seed=8518093/Tue May  9 14:46:22 2023",
        "mu_explore_seed=562691/Tue May  9 14:47:33 2023",
        "mu_explore_seed=6420563/Tue May  9 17:29:44 2023"
    ]

    counts_list_explore = []
    visited_states_list_explore = []
    steps_for_states_visited_list_explore = []  #
    average_test_return_list_explore = []
    training_steps_list_explore = []
    labels_explore = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_explore.append(
            np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

    results_explore = [[returns, steps] for returns, steps in
                       zip(average_test_return_list_explore, training_steps_list_explore)]  # Of form [[ys, xs], ...]
    experiment_names_explore = ["E-MCTS"]
    results_explore_exploration = [[returns, steps] for returns, steps in
                                   zip(visited_states_list_explore, steps_for_states_visited_list_explore)]

    # For learned_ube
    main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/MuUBE/deep_sea/15/"
    indiv_paths = [
        "deep_exploration_no_muexp_seed=3242166/Tue May  9 15:25:01 2023",
    ]

    counts_list_ube = []
    visited_states_list_ube = []
    steps_for_states_visited_list_ube = []  #
    average_test_return_list_ube = []
    training_steps_list_ube = []
    labels_ube = []

    # load results for each seed
    for path in indiv_paths:
        local_path = main_path + path

        counts_list_ube.append(np.load(local_path + "/s_counts.npy"))  # s_counts_at_step_20000 s_counts_at_step_10000

        visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
        steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

        average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
        training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

    results_ube = [[returns, steps] for returns, steps in
                   zip(average_test_return_list_ube, training_steps_list_ube)]  # Of form [[ys, xs], ...]
    experiment_names_ube = ["Only UBE"]
    results_ube_exploration = [[returns, steps] for returns, steps in
                               zip(visited_states_list_ube, steps_for_states_visited_list_ube)]

    # Plot against each other
    results_return = [results_explore, results_ube]
    experiment_names_return = [experiment_names_explore[0], experiment_names_ube[0]]
    results_exploration = [results_explore_exploration, results_ube_exploration]
    experiment_names_exploration = [experiment_names_explore[0], experiment_names_ube[0]]
    titles = ["Average return per step, abstracted model experiments", "Average states visited so far per step, abstracted model experiments"]

    return titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration

def load_final_results():
    subplot_titles = []  # List of shape [plots, 2]
    final_results_return = []  # List of shape [plots, ...]
    final_experiment_names_returns = []  # List of shape [plots, ...]
    final_results_exploration = []  # List of shape [plots, ...]
    final_experiment_names_exploration = []  # List of shape [plots, ...]

    ### Learned Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_learned_model()
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    ### True Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_true_model()
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    ### MuZero Model
    titles, results_return, experiment_names_return, results_exploration, experiment_names_exploration = load_results_muzero_model()
    subplot_titles.append(titles)
    final_results_return.append(results_return)
    final_results_exploration.append(results_exploration)
    final_experiment_names_returns.append(experiment_names_return)
    final_experiment_names_exploration.append(experiment_names_exploration)

    return subplot_titles, final_results_return, final_experiment_names_returns, final_results_exploration, final_experiment_names_exploration


def plot_joined():
    subplot_titles, results_return, experiment_names_returns, results_exploration, experiment_names_exploration = load_final_results()
    sem = True
    plt.figure(0)
    index = 0
    num_plots = 6
    subplot_shape = 3 * 100 + 2 * 10 + index + 1

    for i in range(int(num_plots / 2)):
        subplot_shape = 3 * 100 + 2 * 10 + index + 1
        plt.subplot(subplot_shape)
        plt.title(subplot_titles[i][0], fontsize=8)
        c_p.plot_log(results_return[i], experiment_names_returns[i], sem=sem)
        plt.legend(fontsize=6)
        plt.subplot(subplot_shape + 1)
        plt.title(subplot_titles[i][1], fontsize=8)
        c_p.plot_exploration(results_exploration[i], experiment_names_exploration[i], sem=sem)
        plt.legend(fontsize=6)
        index += 2

    plt.tight_layout()


# Keeps track of recent local experiment
path = "/home/yaniv/EfficientExplore/results/deep_sea/MuMCTSE/deep_sea/15/mu_explore_seed=7038764/Tue May  9 23:14:33 2023/s_counts.npy"  # _at_step_5000 _at_step_10000
counts = np.load(path)
c_p.plot_heat_maps(counts, sa_counts=None, count_cap=20)
# plt.show()

# Take 1
# plot_learned_model_results()
# plot_old_true_model_results()

# Take 2
# plot_new_true_model_results()
# plot_muzero_model()
# plot_muzero_model_2()
# plot_learned_mctse_2()
# plot_true_model_2()

# Take 3
# new_results_learned_model()
# new_results_true_model()
# new_results_muzero_model()

# Take 3, joined
plot_joined()

plt.show()
exit()
#################################
# Load all 3 seeds:
main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/muexplore/deep_sea/15/"
indiv_paths = ["mu_explore_seed=4333300/Thu Apr 27 17:24:43 2023", "mu_explore_seed=2455675/Thu Apr 27 17:25:29 2023",
               "mu_explore_seed=6274682/Thu Apr 27 17:27:24 2023"]

counts_list = []
visited_states_list = []
steps_for_states_visited_list = []  #
average_test_return_list = []
training_steps_list = []
labels = []

# load results for each seed
for path in indiv_paths:
    local_path = main_path + path

    counts_list.append(np.load(local_path + "/s_counts_at_step_10000.npy"))

    visited_states_list.append(np.load(local_path + "/states_visited_per_step.npy"))
    steps_for_states_visited_list.append(np.load(local_path + "/steps_for_states_visited.npy"))

    average_test_return_list.append(np.load(local_path + "/mean_test_results.npy"))
    training_steps_list.append(np.load(local_path + "/training_steps.npy"))

# Plot the exploration rate
plt.gcf()
plt.figure()
for i in range(len(indiv_paths)):
    steps = steps_for_states_visited_list[i]
    visited_states = visited_states_list[i]
    plt.plot(steps, visited_states, label=f"agent {i}")
plt.legend()
plt.title(f"Unique states visited per time step")

min_len = np.min([len(visited_states) for visited_states in visited_states_list])

average_exploration_rate = (visited_states_list[0][:min_len] + visited_states_list[1][:min_len] + visited_states_list[
                                                                                                      2][:min_len]) / 3

plt.gcf()
plt.figure()
steps = steps_for_states_visited_list[0]
plt.plot(steps, average_exploration_rate)
plt.title(f"Unique states visited per time step, average")

# Plot the average heat_map
counts_average = (counts_list[0] + counts_list[1] + counts_list[2]) / 3
c_p.plot_heat_maps(counts_average, sa_counts=None, count_cap=20)

# Plot the test return
results = [[[returns, steps] for returns, steps in
            zip(average_test_return_list, training_steps_list)]]  # Of form [[ys, xs], ...]
experiment_names = ["mu_explore_counts"]
sem = True

plt.gcf()
plt.figure()
plt.title("Average return", fontsize=8)
c_p.plot_log(results, experiment_names, sem=sem)
plt.legend(fontsize=6)

plt.show()
exit()
