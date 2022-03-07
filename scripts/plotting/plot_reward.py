import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from typing import Callable, Tuple
import pandas as pd
import seaborn as sns
from stable_baselines3.common.monitor import load_results


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results_all(log_folder1, log_folder2, log_folder3, save_name, type='timesteps', title='PPO Hovering Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x1, y1 = ts2xy(load_results(log_folder1), type)
    y1 = moving_average(y1, window=44)
    # Truncate x
    x1 = x1[len(x1) - len(y1):]

    x2, y2 = ts2xy(load_results(log_folder2), type)
    y2 = moving_average(y2, window=68)
    # Truncate x
    x2 = x2[len(x2) - len(y2):]

    x3, y3 = ts2xy(load_results(log_folder3), type)
    y3 = moving_average(y3, window=23)
    # Truncate x
    x3 = x3[len(x3) - len(y3):]

    df1 = pd.DataFrame()
    df1['index'] = x1
    df1['reward'] = y1

    df2 = pd.DataFrame()
    df2['index'] = x1
    df2['reward'] = y2

    df3 = pd.DataFrame()
    df3['index'] = x1
    df3['reward'] = y3

    df = pd.concat([df1, df2, df3], ignore_index=True)

    plt.figure(title)
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x="index", y="reward", linewidth=2)# Think of expressing this in millions
    plt.xlabel('Number of ' + type)
    plt.ylabel('Cumulated Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('/home/ubuntu/Plots/' + save_name + '.pdf', format='pdf')
    plt.show()

def plot_results3(log_folder1, lg1, log_folder2, lg2, log_folder3, lg3, save_name, type='timesteps', title='PPO Hovering Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x1, y1 = ts2xy(load_results(log_folder1), type)
    y1 = moving_average(y1, window=50)
    # Truncate x
    x1 = x1[len(x1) - len(y1):]

    x2, y2 = ts2xy(load_results(log_folder2), type)
    y2 = moving_average(y2, window=50)
    # Truncate x
    x2 = x2[len(x2) - len(y2):]

    x3, y3 = ts2xy(load_results(log_folder3), type)
    y3 = moving_average(y3, window=50)
    # Truncate x
    x3 = x3[len(x3) - len(y3):]

    fig = plt.figure(title)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x1, y1, label=lg1)
    sns.lineplot(x2, y2, label=lg2)
    sns.lineplot(x3, y3, label=lg3)
    plt.legend()
    plt.xlabel('Number of '+type)
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('/home/ubuntu/Plots/'+ save_name + type +'.eps', format='eps')
    plt.show()



def plot_results1(log_folder1, lg1, save_name, type='timesteps', title='PPO Hovering Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x1, y1 = ts2xy(load_results(log_folder1), type)
    y1 = moving_average(y1, window=20)
    # Truncate x
    x1 = x1[len(x1) - len(y1):]

    fig = plt.figure(title)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x1, y1, label=lg1)
    plt.legend()
    plt.xlabel('Number of '+type)
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('/home/ubuntu/Plots/'+ save_name + type +'.eps', format='eps')
    plt.show()

def plot_results2(log_folder1, lg1, log_folder2, lg2, save_name, type='timesteps', title='PPO Hovering Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x1, y1 = ts2xy(load_results(log_folder1), type)
    y1 = moving_average(y1, window=50)
    # Truncate x
    x1 = x1[len(x1) - len(y1):]

    x2, y2 = ts2xy(load_results(log_folder2), type)
    y2 = moving_average(y2, window=50)
    # Truncate x
    x2 = x2[len(x2) - len(y2):]

    fig = plt.figure(title)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x1, y1, label=lg1)
    sns.lineplot(x2, y2, label=lg2)
    # sns.lineplot(x3, y3, label=lg3)
    plt.legend()
    plt.xlabel('Number of '+type)
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('/home/ubuntu/Plots/'+ save_name + type +'.eps', format='eps')
    plt.show()

def plot_results4(log_folder1, lg1, log_folder2, lg2, log_folder3, lg3, log_folder4, lg4, save_name, type='timesteps', title='PPO Hovering Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x1, y1 = ts2xy(load_results(log_folder1), type)
    y1 = moving_average(y1, window=50)
    # Truncate x
    x1 = x1[len(x1) - len(y1):]

    x2, y2 = ts2xy(load_results(log_folder2), type)
    y2 = moving_average(y2, window=50)
    # Truncate x
    x2 = x2[len(x2) - len(y2):]

    x3, y3 = ts2xy(load_results(log_folder3), type)
    y3 = moving_average(y3, window=50)
    # Truncate x
    x3 = x3[len(x3) - len(y3):]

    x4, y4 = ts2xy(load_results(log_folder4), type)
    y4 = moving_average(y4, window=50)
    # Truncate x
    x4 = x4[len(x4) - len(y4):]

    fig = plt.figure(title)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x1, y1, label=lg1)
    sns.lineplot(x2, y2, label=lg2)
    sns.lineplot(x3, y3, label=lg3)
    sns.lineplot(x4, y4, label=lg4)
    plt.legend()
    plt.xlabel('Number of '+type)
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('/home/ubuntu/Plots/'+ save_name + type +'.eps', format='eps')
    plt.show()

# Data log directory
# log_dir1 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_3/" # 64 5e-5 std=-0.5
# log_dir2 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_4/" # 128 5e-5 std=-0.5
# log_dir3 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_5/" # 256 5e-5 std=-0.5

# log_dir1 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_7/" # 64 5e-5 std=-1
# log_dir2 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_8/" # 64 5e-5 std=-1
# log_dir3 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_9/" # 64 5e-5 std=-1

# sns.set_style("darkgrid")
#
# data_frame1 = load_results(log_dir1)
# data_frame2 = load_results(log_dir2)
# data_frame3 = load_results(log_dir3)

# plot_results_all(log_dir1, log_dir2, log_dir3, "hovering", type='timesteps')

# plot_results3(log_dir1, "net_arch [64,64]", log_dir2, "net_arch [128,128]", log_dir3, "net_arch [128,128]", "net_arch_diff_init", type='episodes')
# plot_results(log_dir3)



# sns.lineplot(data_frame1)
# fig = plt.figure
# r1 = data_frame1['r']
# r2 = data_frame2['r']
# data_frame2['r1'] = r1
# think about this, doesnt have to be super fancy
# sns.lineplot(x='index', y='r1', data=data_frame2)
# plt.show()
# results_plotter.plot_results([log_dir1], 1e6, results_plotter.X_EPISODES, "PPO Hovering")
# results_plotter.plot_results([log_dir2], 1e6, results_plotter.X_EPISODES, "PPO Hovering")
# results_plotter.plot_results([log_dir3], 1e6, results_plotter.X_EPISODES, "PPO Hovering")
# plt.show()
# plot_results(log_dir1)
# plot_results(log_dir2)

# monitor_1 = pd.read_csv(log_dir+"monitor.csv")
# print(monitor_1)
# plt.show()
#

# Using seaborn with multiple monitor results


# x_var = np.cumsum(monitor_1['l'])
# print(x_var)
#
#
#
#
#
# epi_rew = pd.DataFrame

log_dir1 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_3/" #
log_dir2 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_4/" #
log_dir3 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_5/" #
# log_dir4 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_20/" #
# plot_results1(log_dir1, "test", "test", type='walltime_hrs')

log_dir1_r1 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_4/" # r1 - sparse reward
log_dir1_r2 = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/baseline/PPO_5/" # r2 - direct

# plot_results3(log_dir1, "net_arch [64,64]", log_dir2, "net_arch [128,128]", log_dir3, "net_arch [256,256]", "net_arch_diff", type='episodes')
# plot_results2(log_dir1_r1, "reward function 1", log_dir1_r2, "reward function 2", "reward_function_diff", type='timesteps')