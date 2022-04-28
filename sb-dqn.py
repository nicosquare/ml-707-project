import os
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure

from dotenv import load_dotenv

from src.envs.p2p import P2P

# Initialize Wandb for logging purposes

# load_dotenv()
# wandb.login(key=str(os.environ.get("WANDB_KEY")))

# wandb.init(project="db-dqn-p2p", entity="shahdhardan")

"""
    Main method definition
"""

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--LearningRate", help="Learning rate of the MLP")
parser.add_argument("-m", "--MiniBatch", help="Mini batch size")

# Read arguments from command line
args = parser.parse_args()

if __name__ == '__main__':

    # Parse the parameters

    # LEARNING_RATE = float(args.LearningRate) if args.LearningRate else 1e-3
    # MINIBATCH_SIZE = int(args.MiniBatch) if args.MiniBatch else 16

    # Monitoring the training
    log_dir = "./dqn_tensorboard/" #make sure that this folder exists in your local machine
    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            if (self.num_timesteps % 100 == 0):
                self.logger.dump(self.num_timesteps)
                reward = self.locals['rewards'][0]
                self.logger.record('reward', reward)
                # loss = self.locals['train/loss']
                # self.logger.record('Loss', loss)
            return True


    env = Monitor(P2P())

    # run = wandb.init(
    #     project='db-dqn-p2p',
    #     entity="shahdhardan",
    #     sync_tensorboard=True,
    #     monitor_gym=True,
    #     save_code=True,
    # )
    

    # model_save_path = f"../experiments/sb_dqn_sb/dqn_mg_/{run.id}"

    model = DQN(
        policy="MlpPolicy",
        env=env,
        target_update_interval=10,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0001,
        learning_starts=5,
        max_grad_norm = 1,
        tau = 0.9,
        batch_size=3,
        device='cuda'
        )
        # parser = argparse.ArgumentParser())

    # parser.add_argument("-l", "--LearningRate", help="Learning rate of the MLP")
    # parser.add_argument("-m", "--MiniBatch", help="Mini batch size")

    # Read arguments from command line
    # args = parser.parse_args()

    new_logger = configure(log_dir, ['csv', 'tensorboard'])
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=24*365 + 1,
        n_eval_episodes=10,
        log_interval=4,
        callback=TensorboardCallback()
        # callback=WandbCallback(
        #     model_save_freq=100,
        #     verbose=1,
        #     gradient_save_freq=10,
        #     model_save_path=model_save_path
        # )
    )
    #After you run the code, you can see the tensorboard logs by command: tensorboard --logdir log_dir


    # run.finish()