import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

progress = pd.read_csv("dqn_tensorboard/progress.csv")

progress_l = progress.dropna(subset=["train/loss"])

loss_values = progress_l["train/loss"]

time_steps = range(len(progress_l))

plt.plot(time_steps, loss_values)
plt.title('Loss Curve - DQN - hyperparameters: steps=100000, lr= 0.001, bs=32, learn_starts/upd_intrv = 100')
plt.xlabel('Time Steps')
plt.ylabel('Loss')
plt.show()

reward = progress["reward"]

time_steps = range(len(progress))

plt.plot(time_steps, reward)
plt.title('Reward - DQN - hyperparameters: steps=100000, lr= 0.001, bs=32, learn_starts/upd_intrv = 100')
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.show()