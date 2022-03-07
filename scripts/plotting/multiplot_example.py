import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

df_motor = pd.read_csv('/home/ubuntu/bagfiles/3r/r2_motor.csv', header=0)
df_odom1 = pd.read_csv('/home/ubuntu/bagfiles/net_arch/PPO3_odom.csv', header=0)
df_odom2 = pd.read_csv('/home/ubuntu/bagfiles/net_arch/PPO4_odom.csv', header=0)
df_odom3 = pd.read_csv('/home/ubuntu/bagfiles/net_arch/PPO5_odom.csv', header=0)
print(df_motor)
print(df_motor.columns.values)
# Set motor_rpm dataframe

time = df_motor['time']
time = np.arange(len(time)) / 100

rpm = df_motor['.angular_velocities']
motor = [eval(ele) for ele in rpm]
x0=[]
x1=[]
x2=[]
x3=[]
for a in motor:
    x0.append(abs(a[0]))
    x1.append(abs(a[1]))
    x2.append(abs(a[2]))
    x3.append(abs(a[3]))

# N = 1000
# x1m = np.convolve(x1, np.ones(N)/N, mode='valid')
# x3m = np.convolve(x3, np.ones(N)/N, mode='valid')
# time_m = time[len(time) - len(x1m):smirk:
# y = df_motor['y']
# z = df_motor['z']
# plot

motor_data = pd.DataFrame(time)
motor_data['motor_0']=x0
motor_data['motor_1']=x1
motor_data['motor_2']=x2
motor_data['motor_3']=x3
motor_data.rename(columns={0: 'time [s]'},inplace=True)

# print(motor_data)
# Set odom dataframe
# print(df_odom1)

odomlist = [df_odom1, df_odom2, df_odom3]
networklist = ['net_arch [64,64]', 'net_arch [128,128]', 'net_arch [256,256]']
a = pd.DataFrame()
for odom, network in zip(odomlist, networklist):

    time = odom['time']
    time = np.arange(len(time)) / 100
    name = [network] * len(time)

    odom_data = pd.DataFrame(time)
    odom_data.rename(columns={0: 'time [s]'},inplace=True)

    odom_data['x [m]'] = odom['.pose.pose.position.x']
    odom_data['y [m]'] = odom['.pose.pose.position.y']
    odom_data['z [m]'] = odom['.pose.pose.position.z']

    odom_data['vx [m/s]'] = odom['.twist.twist.linear.x']
    odom_data['vy [m/s]'] = odom['.twist.twist.linear.y']
    odom_data['vz [m/s]'] = odom['.twist.twist.linear.z']

    odom_data['droll [rad/s]'] = odom['.twist.twist.angular.x']
    odom_data['dpitch [rad/s]'] = odom['.twist.twist.angular.y']
    odom_data['dyaw [rad/s]'] = odom['.twist.twist.angular.z']

    odom_data['name'] = name

    b = odom_data
    a = pd.concat([a, b], ignore_index=True)

odom_df = a
print(odom_df)

def plot_multi_xyz(data, x, y, z):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(15,10))
    fig.suptitle('Position')
    axs[0].set_ylim(top=0.1, bottom=-0.1)
    axs[1].set_ylim(top=0.1, bottom=-0.1)
    axs[2].set_ylim(top=12.1, bottom=12-0.1)
    sns.lineplot(x=data.columns[0], y=x, data=data, ax=axs[0], hue="name")
    sns.lineplot(x=data.columns[0], y=y, data=data, ax=axs[1], hue="name")
    sns.lineplot(x=data.columns[0], y=z, data=data, ax=axs[2], hue="name")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axs[0].get_legend().remove()
    axs[2].get_legend().remove()
    fig.tight_layout(pad=3.0)
    plt.show()
    fig.savefig('/home/ubuntu/Plots/arch/position.eps', format='eps')

def plot_multi_vxyz(data, vx, vy, vz):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(15,10))
    fig.suptitle('Velocity')
    axs[0].set_ylim(top=1, bottom=-1)
    axs[1].set_ylim(top=1, bottom=-1)
    axs[2].set_ylim(top=1, bottom=-1)
    sns.lineplot(x=data.columns[0], y=vx, data=data, ax=axs[0], hue="name")
    sns.lineplot(x=data.columns[0], y=vy, data=data, ax=axs[1], hue="name")
    sns.lineplot(x=data.columns[0], y=vz, data=data, ax=axs[2], hue="name")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axs[0].get_legend().remove()
    axs[2].get_legend().remove()
    fig.tight_layout(pad=3.0)
    plt.show()
    fig.savefig('/home/ubuntu/Plots/arch/velocity.eps', format='eps')

def plot_multi_rpy(data, r, p, y):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(15,10))
    fig.suptitle('Angular velocities')
    axs[0].set_ylim(top=10, bottom=-10)
    axs[1].set_ylim(top=10, bottom=-10)
    axs[2].set_ylim(top=2, bottom=-2)
    sns.lineplot(x=data.columns[0], y=r, data=data, ax=axs[0], hue="name")
    sns.lineplot(x=data.columns[0], y=p, data=data, ax=axs[1], hue="name")
    sns.lineplot(x=data.columns[0], y=y, data=data, ax=axs[2], hue="name")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axs[0].get_legend().remove()
    axs[2].get_legend().remove()
    fig.tight_layout(pad=3.0)
    plt.show()
    fig.savefig('/home/ubuntu/Plots/arch/ang_vel.eps', format='eps')

# df_motor2 = pd.DataFrame(time_m)
# df_motor2['x1m']=x1m
# df_motor2['x3m']=x3m
# df_motor2.rename(columns={0: 'time_m'},inplace=True)
# sns.set_theme(style="darkgrid")
#
# #
# sns.lineplot(x="time", y="x1", data=df_motor1)
# z1 = np.polyfit(time, x1, 1)
# p1 = np.poly1d(z1)
# z3 = np.polyfit(time, x3, 1)
# p3 = np.poly1d(z3)
# plt.plot(time,p1(time),"r-")
# plt.plot(time,p3(time),"g-")
# plt.show()
# plt.plot(time_m, x1m)
# plt.plot(time_m, x3m)
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Test')
# # plt.legend([df_motor.columns[1], df_motor.columns[2], df_motor.columns[3]])
# # beautify the x-labels
#
# #
# plt.gcf().autofmt_xdate()
#
# #
# plt.show()

def plot_xys(data, x, y1, y2, y3, y4):
    sns.set_theme(style="darkgrid")
    fig = sns.lineplot(x=x, y=y1, data=data, label='m0')
    sns.lineplot(x=x, y=y2, data=data, label='m1')
    sns.lineplot(x=x, y=y3, data=data, label='m2')
    sns.lineplot(x=x, y=y4, data=data, label='m3')
    plt.hlines(y=[457 + 457, 457 - 457], xmin=[0], xmax=[10], colors='purple', linestyles='--', lw=2)
    fig.set_ylim(top=1000, bottom=-50)
    plt.legend()
    plt.savefig('/home/ubuntu/Plots/motor_speeds.eps', format='eps')
    plt.show()

def plot_xy(data, x, y):
    sns.set_theme(style="darkgrid")
    fig = sns.lineplot(x=x, y=y, data=data, label='motor')
    plt.hlines(y=[457 + 457, 457 - 457], xmin=[0], xmax=[10], colors='purple', linestyles='--', lw=2)
    fig.set_ylim(top=1000, bottom=-50)
    plt.legend()
    plt.savefig('/home/ubuntu/Plots/motor_speed.eps', format='eps')
    plt.show()

# plot_xys(motor_data, 'time [s]', 'motor_0', 'motor_1', 'motor_2', 'motor_3')
# plot_xy(motor_data, 'time [s]', 'motor_0')

plot_multi_xyz(odom_df, 'x [m]', 'y [m]', 'z [m]')
plot_multi_rpy(odom_df, 'droll [rad/s]', 'dpitch [rad/s]', 'dyaw [rad/s]')
plot_multi_vxyz(odom_df, 'vx [m/s]', 'vy [m/s]', 'vz [m/s]')