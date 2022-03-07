# '''
# Q-learning approach for different RL problems
# as part of the basic series on reinforcement learning @
# https://github.com/vmayoral/basic_reinforcement_learning
#
# Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
#
#         @author: Victor Mayoral Vilches <victor@erlerobotics.com>
# '''
# import random
#
# class QLearn:
#     def __init__(self, actions, epsilon, alpha, gamma):
#         self.q = {}
#         self.epsilon = epsilon  # exploration constant
#         self.alpha = alpha      # discount constant
#         self.gamma = gamma      # discount factor
#         self.actions = actions
#
#     def getQ(self, state, action):
#         return self.q.get((state, action), 0.0)
#
#     def learnQ(self, state, action, reward, value):
#         '''
#         Q-learning:
#             Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
#         '''
#         oldv = self.q.get((state, action), None)
#         if oldv is None:
#             self.q[(state, action)] = reward
#         else:
#             self.q[(state, action)] = oldv + self.alpha * (value - oldv)
#
#     def chooseAction(self, state, return_q=False):
#         q = [self.getQ(state, a) for a in self.actions]
#         maxQ = max(q)
#
#         if random.random() < self.epsilon:
#             minQ = min(q); mag = max(abs(minQ), abs(maxQ))
#             # add random values to all the actions, recalculate maxQ
#             q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
#             maxQ = max(q)
#
#         count = q.count(maxQ)
#         # In case there're several state-action max values
#         # we select a random one among them
#         if count > 1:
#             best = [i for i in range(len(self.actions)) if q[i] == maxQ]
#             i = random.choice(best)
#         else:
#             i = q.index(maxQ)
#
#         action = self.actions[i]
#         if return_q: # if they want it, give it!
#             return action, q
#         return action
#
#     def learn(self, state1, action1, reward, state2):
#         maxqnew = max([self.getQ(state2, a) for a in self.actions])
#         self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

import numpy as np
import matplotlib.pyplot as plt
import random

# state = []
# value = []
# for a in range(50):
#     state.append(random.uniform(-5,5))
#
# for x in state:
#     value.append(x**3-2*x**2-3*x+4 + random.uniform(0, 10))
#
# coefficients = np.polyfit(state, value, 4)
# poly = np.poly1d(coefficients)
#
# new_x = np.linspace(-5, 5, 512, endpoint = True)
# new_y = poly(new_x)
#
# a = 1
# b = 2
# c = 3
# d = 4
# x = np.linspace(-5, 5, 512, endpoint = True)
# y = a * (x**3) - b * x**2 - c * x + d
#
# plt.plot(x, y, '-g', label="True F(x)")
# plt.plot(new_x, new_y, '-r', label="f(x)")
# plt.scatter(state, value, label="Data")
# axes = plt.gca()
# axes.set_xlim([x.min(), x.max()])
# axes.set_ylim([y.min(), y.max()])
#
# plt.xlabel('state',fontsize=15)
# plt.ylabel('value', fontsize=15 )
# plt.title('Function', fontsize=20)
# plt.legend(loc='lower right', prop={'size': 15})
#
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def tanh(x):
#     t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#     dt=1-t**2
#     return t,dt
# z=np.arange(-4,4,0.01)
# tanh(z)[0].size,tanh(z)[1].size
#
# # Setup centered axes
#
# sns.set_style("darkgrid")
# fig, ax = plt.subplots(figsize=(9, 5))
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# # Create and show plot
# ax.plot(z,tanh(z)[0], color="#307EC7", linewidth=3, label="tanh")
# ax.plot(z,tanh(z)[1], color="#9621E2", linewidth=3, label="derivative")
# ax.legend(loc="lower right", frameon=False)
# plt.title("Tanh")
# fig.show()
# plt.savefig('/home/ubuntu/Plots/tanhd.eps', format='eps')
# plt.show()


x = np.arange(-4,4,0.01)
y = np.maximum(0, x)
def f(x):
    if(x < 0): return 0
    else: return 1

z = []
for i in range(len(x)):
   z.append(f(x[i]))

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(x, y, color="#307EC7", linewidth=3, label="ReLU")
ax.plot(x, z, color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="lower right", frameon=False)
plt.title("ReLU")
fig.show()
plt.savefig('/home/ubuntu/Plots/relu.eps', format='eps')
plt.show()
