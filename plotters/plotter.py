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


def load_counts_and_exploration_rate():
    pass




"""
    This script is used to load and plot the final results.
"""

# Keeps track of recent local experiment
path = "/home/yaniv/EfficientExplore/results/deep_sea/MuMCTSE/deep_sea/10/mu_explore_seed=8244051/Mon May  1 10:11:58 2023/s_counts.npy"  # _at_step_5000 _at_step_10000
counts = np.load(path)
c_p.plot_heat_maps(counts, sa_counts=None, count_cap=20)
# plt.show()

# Load all alpha_explore seeds that worked
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

    counts_list_explore.append(np.load(local_path + "/s_counts_at_step_20000.npy"))

    visited_states_list_explore.append(np.load(local_path + "/states_visited_per_step.npy"))
    steps_for_states_visited_list_explore.append(np.load(local_path + "/steps_for_states_visited.npy"))

    average_test_return_list_explore.append(np.load(local_path + "/mean_test_results.npy"))
    training_steps_list_explore.append(np.load(local_path + "/training_steps.npy"))

results_explore = [[[returns, steps] for returns, steps in zip(average_test_return_list_explore, training_steps_list_explore)]]    # Of form [[ys, xs], ...]
experiment_names_explore = ["AlphaExplore"]

# Load all alpha_ube seeds that worked
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

    counts_list_ube.append(np.load(local_path + "/s_counts_at_step_20000.npy"))

    visited_states_list_ube.append(np.load(local_path + "/states_visited_per_step.npy"))
    steps_for_states_visited_list_ube.append(np.load(local_path + "/steps_for_states_visited.npy"))

    average_test_return_list_ube.append(np.load(local_path + "/mean_test_results.npy"))
    training_steps_list_ube.append(np.load(local_path + "/training_steps.npy"))

results_ube = [[[returns, steps] for returns, steps in zip(average_test_return_list_ube, training_steps_list_ube)]]    # Of form [[ys, xs], ...]
experiment_names_ube = ["AlphaUBE"]

# Plot against each other
sem = True
plt.gcf()
plt.figure()
plt.title("Average return", fontsize=8)
c_p.plot_log(results_explore, experiment_names_explore, sem=sem)
c_p.plot_log(results_ube, experiment_names_ube, sem=sem)
plt.legend(fontsize=6)

plt.show()
exit()

#################################
# Load all 3 seeds:
main_path = "/home/yaniv/EfficientExplore/results_from_cluster/results/deep_sea/muexplore/deep_sea/15/"
indiv_paths = ["mu_explore_seed=4333300/Thu Apr 27 17:24:43 2023", "mu_explore_seed=2455675/Thu Apr 27 17:25:29 2023", "mu_explore_seed=6274682/Thu Apr 27 17:27:24 2023"]

counts_list = []
visited_states_list = []
steps_for_states_visited_list = []  #
average_test_return_list = []
training_steps_list = []
labels = []

# load results for each seed
for path in indiv_paths:
    local_path = main_path + path

    counts_list.append(np.load(local_path + "/s_counts_at_step_20000.npy"))

    visited_states_list.append(np.load(local_path + "/states_visited_per_step.npy"))
    steps_for_states_visited_list.append(np.load(local_path + "/steps_for_states_visited.npy"))

    average_test_return_list.append(np.load(local_path + "/mean_test_results.npy"))
    training_steps_list.append(np.load(local_path + "/training_steps.npy"))

# Plot the exploration rate
plt.figure(0)
for i in range(len(indiv_paths)):
    steps = steps_for_states_visited_list[i]
    visited_states = visited_states_list[i]
    plt.plot(steps, visited_states, label=f"agent {i}")
plt.legend()
plt.title(f"Unique states visited per time step")

min_len = np.min([len(visited_states) for visited_states in visited_states_list])

average_exploration_rate = (visited_states_list[0][:min_len] + visited_states_list[1][:min_len] + visited_states_list[2][:min_len]) / 3

plt.figure(1)
steps = steps_for_states_visited_list[0]
plt.plot(steps, average_exploration_rate)
plt.title(f"Unique states visited per time step, average")

# Plot the average heat_map
counts_average = (counts_list[0] + counts_list[1] + counts_list[2]) / 3
c_p.plot_heat_maps(counts_average, sa_counts=None, count_cap=20)

# Plot the test return
results = [[[returns, steps] for returns, steps in zip(average_test_return_list, training_steps_list)]]    # Of form [[ys, xs], ...]
experiment_names = ["mu_explore_counts"]
sem = True

plt.gcf()
plt.figure()
plt.title("Average return", fontsize=8)
c_p.plot_log(results, experiment_names, sem=sem)
plt.legend(fontsize=6)


plt.show()
exit()


