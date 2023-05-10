from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
import pathlib


def get_color_names(all_colors=False):
    colors = []
    if all_colors:
        for name, hex in mcolors.cnames.items():
            colors.append(name)
    else:
        colors = ["blue", "red", "#54278f", "orange", "green", "teal", "magenta"]
        colors = ["#332288", "#88CCEE",
                  "#f1a340", # "#44AA99",
                  "#117733",
                  "#999933", "#DDCC77",
                  "#CC6677", "#882255", "#AA4499"
                  ]
        # This is a list of other colors, in hexa
        # colors = ["#762a83", "#1b7837", "#f1a340", "#4575b4", "#31a354", "#045a8d"]
    return colors


def plot_log(results, experiment_names, index=0, opt=None, sem=False):
    colors = get_color_names()
    # Standard error of th mean or STD?
    for i in range(len(results)):
        data = results[i]
        max_epi = np.min([len(s[-1]) for s in data])
        # Bins:
        # tried 100, 50, 30, 20. For mountaincar 30 looked best (smooth enough, junky enough to be believable).
        # For slide, can go up to 100
        bins = 30
        x, m, s = smoothed_statistics([s[-1] for s in data], [s[index] for s in data], bins=bins, sem=sem)
        if len(data) > 1:
            plt.fill_between(x, [m[t] - s[t] for t in range(len(m))],
                            [m[t] + s[t] for t in range(len(m))], color=colors[i], alpha=0.2)
        plt.plot(x, m, colors[i], label=experiment_names[i])
        plt.xlabel('Environmental steps', fontsize=8)   # Was 8 for separate experiments
        plt.ylabel('Episodic return', fontsize=8)       # Was 8 for separate experiments
    if opt is not None:
        plt.plot([x[0], x[-1]], [opt, opt], ':k')

def plot_exploration(results, experiment_names, index=0, opt=None, sem=False):
    colors = get_color_names()
    # Standard error of th mean or STD?
    for i in range(len(results)):
        data = results[i]
        max_epi = np.min([len(s[-1]) for s in data])
        # Bins:
        # tried 100, 50, 30, 20. For mountaincar 30 looked best (smooth enough, junky enough to be believable).
        # For slide, can go up to 100
        bins = 30
        x, m, s = smoothed_statistics([s[-1] for s in data], [s[index] for s in data], bins=bins, sem=sem)
        if len(data) > 1:
            plt.fill_between(x, [m[t] - s[t] for t in range(len(m))],
                            [m[t] + s[t] for t in range(len(m))], color=colors[i], alpha=0.2)
        plt.plot(x, m, colors[i], label=experiment_names[i])
        plt.xlabel('Environmental steps', fontsize=8)   # Was 8 for separate experiments
        plt.ylabel('Explored states', fontsize=8)       # Was 8 for separate experiments
    if opt is not None:
        plt.plot([x[0], x[-1]], [opt, opt], ':k')

def repair_time_series(x: list):
    """ Linearly interpolates a time-series with missing (nan) values.
        First and last entry cannot be interpolated. """
    for i in range(1, len(x) - 1):
        if np.isnan(x[i]):
            n = 1
            if i < len(x) - 1:
                j = i + 1
                while j < len(x) - 1 and np.isnan(x[j]):
                    n += 1
                    j += 1
            for j in range(n):
                x[i + j] = (n - j) * x[i - 1] / (n + 1) + (j + 1) * x[i + n] / (n + 1)
    return x


def history_smoothing(x, y, max_x=None, min_x=0.0, bins=100):
    """ Uses history smoothing (w.r.t. to the x parameter) with a given number of bins on the y parameters. """
    x = x[(len(x) - len(y)):]
    max_x = np.max(x) if max_x is None else max_x
    hn = [0 for _ in range(bins)]
    hy = [0 for _ in range(bins)]
    for t in range(len(x)):
        i = min(int((x[t] - min_x) / (max_x - min_x) * bins), bins - 1)
        hn[i] += 1
        hy[i] += y[t]
    return repair_time_series([(hy[i] / hn[i]) if hn[i] > 0 else float('nan') for i in range(bins)])


def smoothed_statistics(xs, ys, bins=100, sem=False):
    """ Returns the history-smoothed x-values, means and standard deviations (or alternatively SEMs)
        of a list of time series (i.e., each time series corresponds to an independent seed). """
    min_x = np.max([np.min(x) for x in xs])
    max_x = np.min([np.max(x) for x in xs])
    hx = [i * (max_x - min_x) / (bins - 1) for i in range(bins)]
    hys = [history_smoothing(xs[i], ys[i], max_x=max_x, min_x=min_x, bins=bins) for i in range(len(xs))]
    my = [np.mean([y[t] for y in hys]) for t in range(bins)]
    sy = [np.std([y[t] for y in hys]) for t in range(bins)]
    if sem: sy = [si / np.sqrt(len(xs)) for si in sy]
    # return hx, my, sy
    return hx[1:], my[1:], sy[1:] # this line for the negative reward experiments, to ignore false first entry in results


def load_results(path, x_name=["num_played_steps.npy", "played_steps.npy", "played.npy"], y_name=["total_rewards.npy", "rewards.npy"]):
    """
        Loads the results for a specific experiment (for example, mountaincar exploratory counter uncertainty)
         with _ seeds (for example, 10) into one list, the way the smoother wants
    """
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


def plot_heat_maps(s_counts, sa_counts=None, count_cap=None):
    """
    Plots a heatmap of the values in the table.
    Parameters:
        s_counts: np array of shape [H, W]
        sa_counts: np array of shape [H, W, 2]
        count_cap: the maximum count we want to present.
    """
    if count_cap is not None:
        s_counts[s_counts > count_cap] = count_cap
        if sa_counts is not None:
            sa_counts[sa_counts > count_cap] = count_cap

    cmap = plt.cm.get_cmap('plasma_r')  # 10 distinct colors
    cmap.set_under('white')  # set color for 0 counts to black

    # Define a color scheme with 10 distinct colors
    fig, ax = plt.subplots()
    im = ax.imshow(s_counts, cmap=cmap, vmin=0.5, vmax=count_cap + 0.5 if count_cap is not None else 10.5)

    # Set up the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    # Set up the x and y axis labels
    ax.set_xticks(np.arange(0, s_counts.shape[1], 5))
    ax.set_yticks(np.arange(0, s_counts.shape[0], 5))
    ax.set_xticklabels(np.arange(0, s_counts.shape[1], 5))
    ax.set_yticklabels(np.arange(0, s_counts.shape[0], 5))

    # Rotate the x-axis labels and set the title
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Heat Map of Counts")

    # Loop over data dimensions and create text annotations.
    # for i in range(s_counts.shape[0]):
    #     for j in range(s_counts.shape[1]):
    #         text = ax.text(j, i, s_counts[i, j],
    #                        ha="center", va="center", color="w")


def plot_scores_vs_counts(s_counts, scores, N, bucket_cap):
    unique_counts = np.unique(s_counts)
    count_buckets = np.arange(0, bucket_cap + 1, N)
    bucket_scores = [[] for _ in range(len(count_buckets))]
    for count in unique_counts:
        count_score = scores[s_counts == count]
        bucket_idx = min(int(count / bucket_cap * len(count_buckets)), len(count_buckets) - 1)
        bucket_scores[bucket_idx].extend(count_score)

    avg_scores = []
    std_scores = []
    for bucket in bucket_scores:
        if len(bucket) > 0:
            avg_score = np.mean(bucket)
            std_score = np.std(bucket)
            avg_scores.append(avg_score)
            std_scores.append(std_score)
        else:
            avg_scores.append(0)
            std_scores.append(0)

    fig, ax = plt.subplots()
    ax.scatter(count_buckets, avg_scores)
    ax.errorbar(count_buckets, avg_scores, yerr=std_scores, fmt='none', capsize=5)
    ax.set_xlabel('Counts')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Score vs. Counts')

# # Code to print the hexas and names of colors
# for name, hex in mcolors.cnames.items():
#     print(name, hex)