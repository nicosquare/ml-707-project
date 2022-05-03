import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

progress = pd.read_csv("dqn_tensorboard/progress.csv")

progress_l = progress.dropna(subset=["train/loss"])

loss_values = progress_l["train/loss"]

# loss_values = pd.read_csv("notebooks/loss_3.csv")["Value"]

time_steps = range(len(loss_values))

plt.plot(time_steps, loss_values)
plt.title('Loss - DQN - hyp: t=100000, lr= 0.001, bs=32, learn_starts/upd_intrv = 100')
plt.xlabel('Time Steps')
plt.ylabel('Loss')
plt.savefig("try2.png")

progress_r = progress.dropna(subset=["reward"])

reward = progress_r["reward"]

# reward = pd.read_csv("notebooks/reward_3.csv")["Value"]

time_steps = range(len(reward))

plt.plot(time_steps, reward)
plt.title('Reward - DQN - hyp: t=100000, lr= 0.001, bs=32, learn_starts/upd_intrv = 100')
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.savefig("try2_reward.png")