# Hummingbird_pkg

**Hummingbird_pkg** is a rospy based reinforcement learning component of the reinforcemencement learning based quadrotor controller training framework.

Hummingbird_pkg cannot be used as a standalone package, it must be running within the pre-settings. The following dependencies are the prerequisites to initiate the training:

- [ROS noetic]
- [RotorS]
- [Openai_ros]
- [Stable baselines3]

## Fault-Tolerant Quadrotor Controller using DRL

For my Master Thesis, I have used **Hummingbird_pkg** to configure the environment settings for my training environment (e.g., terminal conditions, observations and actions). The schematics of the overall training environment is shown below:

<img src="/assets/img/env.png" alt="Traning Environment" style="height: 500px; width:900px;"/>

The trained controller can adjust itself to single rotor failure while hovering:

<img src="assets/img/t5.gif" width="400" height="400" />

Please refer to the thesis for further details:

If you are willing to cite this research for your publication, please cite:

```bibtex
@masterthesis{Kim2022,
    author = {Kim, Taehyoung and Armanini, Sophie},
    school = {Technical University of Munich}, 
    title = {Fault-Tolerant Quadrotor Controller Design Using Deep Reinforcement Learning},
    year = {2022},
}
```

If this project was helpful, don't hesitate to give this repo a star :D

## Dependencies Installation

# ROS / ROTORS

1. Install ROS initialize ROS noetic desktop full, additional ROS packages, catkin-tools, and wstool:

```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install ros-noetic-desktop-full ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavlink python3-wstool python3-catkin-tools protobuf-compiler libgoogle-glog-dev ros-noetic-control-toolbox ros-noetic-mavros
sudo apt-get install python3-pip
sudo pip install -U rosdep
sudo rosdep init
rosdep update
source /opt/ros/noetic/setup.bash
```

2. If you don't have ROS workspace set one:

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace  # initialize your catkin workspace
wstool init
```

3. Get the simulator and additional dependencies

```
cd ~/catkin_ws/src
git clone https://github.com/ethz-asl/rotors_simulator.git
git clone https://github.com/ethz-asl/mav_comm.git
```

3. Build your ws

```
cd ~/catkin_ws/
catkin init  # If you haven't done this before.
catkin build
```

# SB3 / OPENAI_ROS
Download and install

```
# SB3
pip install stable-baselines3[extra]
# OPENAI_ROS
git clone -b version2 https://bitbucket.org/theconstructcore/openai_ros.git
```
> **Note** For the openai_ros package, don't forget to build it on your ws

> **Note** Please add self.step_counter = 0 in robot_gazebo_env.py in Openai_ros

## Hummingbird_pkg Installation
Please clone the repo and install it on your ws and build it:

```
git clone https://github.com/GOMTAE/hummingbird_pkg.git
```

## Hummingbird_pkg Setup
Please make sure that all necessary directories for the operation are specified correctly. If not, please change it.
The files that you should change the directories are:
- Config files in **config** folder (param.yaml file e.g. hummingbird_ppo_params_baseline.yaml)
- evaluation scripts in **scripts** folder (e.g. hummingbird_eval_baseline.py)
- training scripts in **scripts** folder (e.g. train_hummingbird_ppo_basline.py)

## Basic Usage
In order to initiate the training, you first have to launch the simulation environment. The simulation environment can be initiated via following: 

```
$ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic
```

We initialize the drone in airborne condition. You can of course change the initial position and the orientation of the drone from the **rotors_gazebo** package.

### Traning the quadrotor to fly
After launching the simulation environment, you can initialte the training
```
roslaunch hummingbird_pkg start_training_gt_ppo_baseline.launch # Train baseline controller on ground truth environment (no IMU noise)
roslaunch hummingbird_pkg start_training_gt_ppo_3rotors.launch  # Continue traning Fault-tolerant case on ground truth environment (no IMU noise)
roslaunch hummingbird_pkg start_training_ppo_baseline.launch    # Train baseline controller on noisy environment (IMU noise)
roslaunch hummingbird_pkg start_training_ppo_3rotors.launch     # Continue traning Fault-tolerant case on noisy environment (IMU noise)
```
Please make sure to change the path before training. (yaml files in the **config** folder for the env configuration, and the training/evaluation scripts in the **scirpts** folder)

### Enjoy the trained agent
The simulation environment must be initiated first. The one of the trained agent models are added for complete user journey.

```
roslaunch hummingbird_pkg gt_eval_baseline.launch # Trained versatile controller which works as a hovering and fault-tolerant controller
roslaunch hummingbird_pkg gt_eval_3rotors.launch  # Trained versatile controller which works as a hovering and fault-tolerant controller
roslaunch hummingbird_pkg eval_baseline.launch    # Hovering only (baseline)
roslaunch hummingbird_pkg eval_3rotors.launch     # Trained versatile controller which works as a hovering and fault-tolerant controller
```

Please modify the path of the models to evaluate your own trained agents.

**Please note**:  There is no parser implementation on our code. Therefore you have to manually change the path of the model you are calling for evaluation.

## Acknowledgement
I want to express my gratitude to many researchers who provided the simulator and package as open-source, making the research possible!

## License
MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
    
   [ROS noetic]: <http://wiki.ros.org/noetic>
   [RotorS]: <https://github.com/ethz-asl/rotors_simulator>
   [Openai_ros]: <https://bitbucket.org/theconstructcore/openai_ros.git>
   [Stable baselines3]: <https://github.com/DLR-RM/stable-baselines3>
