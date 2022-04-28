import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

loss_values = pd.read_csv("dqn_tensorboard/progress.csv")["train/loss"]

plt.plot(loss_values)
plt.title('Loss Curve - DQN - hyperparameters: ')
plt.xlabel('Time Steps')
plt.ylabel('Loss')
plt.show()