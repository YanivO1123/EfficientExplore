from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from plotters import plotting_utils as c_p
import os
import pathlib
import re


"""
    This file is used to load and plot the final results.
"""


def cleanup(top_path="../final_results/"):
    """
        This function deletes all the necessary files from the final-results folder, so that they become
        reasonably-sized and uploadeble to git
    """
    # Make a list of the different seeds' folder names
    for path_name in os.listdir(top_path):
        path = top_path + path_name + "/"
        dir_names = [sorted([path + top_dir_name + "/" + dir_name for dir_name in os.listdir(path + top_dir_name)]) for
                     top_dir_name in os.listdir(path)]

        # Make a list of the separate files
        to_remove_files = [
            [
                [
                    dir_name + "/" + filename
                    for filename in sorted(
                    os.listdir(dir_name)
                )
                    if not filename.endswith(".npy") and "config" not in filename
                ] for dir_name in top_dir_names
            ]
            for top_dir_names in dir_names
        ]

        # Remove all not-my-results files, to save space
        for top_dir in to_remove_files:
            for innder_dir in top_dir:
                for file in innder_dir:
                    print(file)
                    os.remove(file)

        # # Rename files from played_steps.npy to played.npy because WSL is retarded
        # if path_name == "mountaincarDoublePlanningAblations":
        #     to_rename_files = [
        #         [
        #             [
        #                 dir_name + "/" + filename
        #                 for filename in sorted(
        #                 os.listdir(dir_name)
        #             )
        #                 if filename.endswith("total_rewards.npy")
        #             ] for dir_name in top_dir_names
        #         ]
        #         for top_dir_names in dir_names
        #     ]
        #
        #     # Rename files
        #     for top_dir in to_rename_files:
        #         for innder_dir in top_dir:
        #             for file_name in innder_dir:
        #                 new_file_name = os.path.dirname(file_name) + "/rewards.npy"
        #                 os.rename(file_name, new_file_name)
        #                 print(new_file_name)


def plot_general(index,
                 subplot_titles,
                 paths_to_experiments,
                 sem=True,
                 plot_seperately=True,  # Plot each experiment in its own figure, or all in the same subplot
                 num_rows=None, # How many expriments (how many subplot rows)
                 ):
    if len(subplot_titles) < 1:
        return index
    elif len(subplot_titles) == 1:
        ### Load first
        results_first_plot = []
        path_first_plot = paths_to_experiments[0]
        path_list_first_plot = [
            path_first_plot +
            dir_name + "/"
            for dir_name in os.listdir(path_first_plot)
        ]
        experiment_names_first_plot = []
        for path in sorted(path_list_first_plot):
            results_first_plot.append(c_p.load_results(path))
            path = pathlib.PurePath(path)
            experiment_names_first_plot.append(path.name)

        # load
        results = []
        path = paths_to_experiments[0]
        individual_paths = [
            path +
            dir_name + "/"
            for dir_name in os.listdir(path)
        ]
        experiment_names = []
        for path in sorted(individual_paths):
            results.append(c_p.load_results(path))
            path = pathlib.PurePath(path)
            experiment_names.append(path.name)
        # for path in sorted(individual_paths):
        # results.append(c_p.load_results(path))
        # path = pathlib.PurePath(path)
        # experiment_names.append(path.name)

        plt.figure(index)
        plt.title(subplot_titles[0], fontsize=8)
        c_p.plot_log(results, experiment_names, sem=sem)
        plt.legend(fontsize=6)

        index += 1

        return index
        # plot
    elif len(subplot_titles) == 2:
        ### Load first
        results_first_plot = []
        path_first_plot = paths_to_experiments[0]
        path_list_first_plot = [
            path_first_plot +
            dir_name + "/"
            for dir_name in os.listdir(path_first_plot)
        ]
        experiment_names_first_plot = []
        for path in sorted(path_list_first_plot):
            results_first_plot.append(c_p.load_results(path))
            path = pathlib.PurePath(path)
            experiment_names_first_plot.append(path.name)

        ### Load second
        results_second_plot = []
        path_second_plot = paths_to_experiments[1]
        path_list_second_plot = [
            path_second_plot +
            dir_name + "/"
            for dir_name in os.listdir(path_second_plot)
        ]
        experiment_names_second_plot = []
        for path in sorted(path_list_second_plot):
            results_second_plot.append(c_p.load_results(path))
            path = pathlib.PurePath(path)
            experiment_names_second_plot.append(path.name)

        if plot_seperately:
            # Plot
            plt.figure(index)
            plt.subplot(121)
            plt.title(subplot_titles[0], fontsize=8)
            c_p.plot_log(results_first_plot, experiment_names_first_plot, sem=sem)
            plt.legend(fontsize=6)
            plt.subplot(122)
            plt.title(subplot_titles[1], fontsize=8)
            c_p.plot_log(results_second_plot, experiment_names_second_plot, sem=sem)
            plt.legend(fontsize=6)
            index += 1
        else:
            plt.figure(0)
            subplot_shape = num_rows * 100 + 2 * 10 + index + 1
            print(subplot_shape)
            plt.subplot(subplot_shape)
            plt.title(subplot_titles[0], fontsize=8)
            c_p.plot_log(results_first_plot, experiment_names_first_plot, sem=sem)
            plt.legend(fontsize=6)
            plt.subplot(subplot_shape + 1)
            plt.title(subplot_titles[1], fontsize=8)
            c_p.plot_log(results_second_plot, experiment_names_second_plot, sem=sem)
            plt.legend(fontsize=6)
            index += 2

        return index
    else:
        print("Only implemented plotting of up to 2 plots")


def load_counts(path):
    pass

# Test heat_map plotter
# Generate some random data

path = "/home/yaniv/EfficientExplore/results/deep_sea/muexplore/deep_sea/7/mu_explore_seed=5339032/Tue Apr 18 19:11:45 2023/s_counts.npy"
# s_counts = np.random.randint(1000, size=(50, 50))
s_counts = np.load(path)
scores = np.random.rand(50, 50)
bucket_size = 2
bucket_cap = 5
# Plot the heat map
c_p.plot_heat_maps(s_counts, sa_counts=None, count_cap=20)
# c_p.plot_scores_vs_counts(s_counts, scores, bucket_size, bucket_cap)
plt.show()
exit()

# control font sizes:
matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)

# # cleanup() will remove all event and replay-buffer files from the final results folder. Actual results are saved.
# cleanup()

# The index makes sure the figures are indexed correctly
index = 0
all_experiment_paths = [
    # ["../final_results/slide_core/", "../final_results/mountaincar_terminal/"],    # This is for mc terms + regular slide
    # ["../final_results/slide_core/", "../final_results/mountaincar_core/"],
    # ["./slideValueTargetAblations/", "../final_results/mountaincarValueTargetAblations/"],
    # ["../final_results/slidePolicyTargetAblations/", "../final_results/mountaincarPolicyTargetAblations/"],
    # ["./slideDoublePlanningAblations/", "./mountaincarDoublePlanningAblations/"],
    # ["../final_results/slideAlternatingLearningAblations/", "./mountaincarAlternatingLearningAblations/"],
    # ["../final_results/slide_terminal/", "../final_results/mountaincar_terminal/"],
    # ["./mountaincarJointCounterCoeffExperiments/", "./mountaincarJointEnsembleCoeffExperiments/"],
    # ["../final_results/mountaincarValueTargetAblations/", "../final_results/mountaincarPolicyTargetAblations/"], # for ablations for the paper
    # ["./mountaincarAlternatingLearningAblations/", "./mountaincarDoublePlanningAblations/"],
    ["./slide_3_actions/"],
]
all_experiment_titles = [
    # ["Results against the Slide environment", "Results against the Mountain Car environment"],
    # ["Results against the Slide environment", "Results against the Mountain Car environment"],
    # ["Slide, value targets ablation study", "Mountain Car, value targets ablation study"],
    # ["Slide, policy targets ablation study", "Mountain Car, policy targets ablation study"],
    # ["Slide, double planning ablation study", "Mountain Car, double planning ablation study"],
    # ["Slide, alternating episodes ablation study", "Mountain Car, alternating episodes ablation study"],
    # ["Slide, negative rewards scheme", "Mountain Car, original reward scheme"],
    # ["Robust-exploratory, visitation counting, growing coeff.", "Robust-exploratory, ensemble, growing coeff."],
    # ["Mountain Car, value targets ablation study", "Mountain Car, policy targets ablation study"], # for ablations for the paper
    # ["Mountain Car, alternating episodes ablation study", "Mountain Car, double planning ablation study"], # for ablations for the paper
    ["Slide, three actions, length 60, starting at position 10"],
]

# This will print everything in the above lists, to plot through one functionality.
# If we wish to plot one specific plot, just comment the other paths and titles in the above lists
plt.figure(index)
for i in range(len(all_experiment_paths)):
    index = plot_general(index, all_experiment_titles[i], all_experiment_paths[i], plot_seperately=True, num_rows=len(all_experiment_titles))

### Print plots
plt.show()
