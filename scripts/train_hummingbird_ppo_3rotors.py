#!/usr/bin/env python
# ROS packages required
import rospy
import rospkg
# Dependencies required
import gym
import os
import numpy as np
import pandas as pd
import time

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3, HER
# from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
# from stable_baselines.sac.policies import MlpPolicy as mlp_sac
# from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
# from stable_baselines.td3.policies import MlpPolicy as mlp_td3
# from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DDPG, PPO, A2C, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# For scheduler
from typing import Callable

# import our task environment
import hummingbird_hover_task_env_ppo_3rotors

# ROS ENV gets started automatically before the training
# from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment -- Not implemented

# change the directory
os.chdir('/home/ubuntu/catkin_ws/src/hummingbird_pkg/') # Change directory
rospy.init_node('hummingbird_training_ppo_3rotors', anonymous=True, log_level=rospy.FATAL)

# check log directory
log_dir = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/3rotors/" # Change directory
os.makedirs(log_dir, exist_ok=True)

# Create the Gym environment
environment_name = rospy.get_param('/hummingbird/task_and_robot_environment_name')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: Monitor(env, log_dir)])
env = VecNormalize.load(log_dir + "PPO_final.pkl", env)

# rospy.loginfo("Gym environment done")
# Set the logging system
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hummingbird_pkg')
outdir = pkg_path + '/training_results'

# Save a checkpoint every 100k steps
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/checkpoint/3rotors/PPO',
                                         name_prefix='ppo_model') # Change directory

# Continue learning for extra 5mil timesteps
TIMESTEPS = 5000000

# Load baseline
baseline = PPO.load(log_dir + "PPO_final", tensorboard_log="/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/tensorboard_logs/PPO/3rotors/") # Change directory
baseline.set_env(env)

# Linear decay your learning rate
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

lr = linear_schedule(0.00005)

# The noise objects for TD3 -- how to implement this on continuous action space?
# n_actions = env.action_space.n
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model_list = [
        # A2C(MlpPolicy, env, verbose=1, tensorboard_log="/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/tensorboard_logs/A2C/"),
        baseline,
        # DDPG(MlpPolicy, env, verbose=1, ent_coef=0.001, learning_rate=0.0005, tensorboard_log="/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/tensorboard_logs/PPO2/"),
        # TD3(MlpPolicy, env, verbose=1, tensorboard_log="/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/tensorboard_logs/TRPO/"),
        # SAC(MlpPolicy, env, verbose=1, tensorboard_log="/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/tensorboard_logs/TRPO/")
]
#
algo_list = [
             # 'A2C',
             'PPO',
             # 'DDPG',
             # 'TD3',
             # 'SAC'
             ]

training_time_list = []
training_hp_list = []
for model, algo in zip(model_list, algo_list):
    print(model)

    # Don't forget to save the VecNormalize statistics when saving the agent
    log_dir = "/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/3rotors/" # Change directory

    start = time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    hp = [model.learning_rate, model.n_steps, model.batch_size, model.policy_kwargs]
    end = time.time()
    training_time_list.append((end-start)*1000)
    training_hp_list.append(hp)

    model.save(log_dir+algo+"_hummingbird_hover_3rotor")
    env.save(log_dir+algo+"_hummingbird_hover_vec_normalize_3rotor.pkl")

df = pd.DataFrame(list(zip(algo_list, training_time_list, training_hp_list)), columns=['algo', 'train_time (ms)', 'hp (total_ts, n_steps, batch_size, policy_kwargs'])
df.to_csv('/home/ubuntu/catkin_ws/src/hummingbird_pkg/results/trained_model/3rotors/train_info.csv', index=False) # Change directory

env.close()