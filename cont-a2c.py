"""

    Advantage Actor Critic (A2C) algorithm implementation

    Credits: Nicolás Cuadrado, MBZUAI, OptMLLab

"""
import os
import numpy as np
import torch
import wandb

from gym import Env
from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.optim import Adam
from torch.distributions import Normal
from dotenv import load_dotenv

from src.components.battery import BatteryParameters
from src.envs.p2p import P2PA2C

torch.set_default_dtype(torch.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize Wandb for logging purposes

load_dotenv()
wandb.login(key=str(os.environ.get("WANDB_KEY")))

# Define global variables

zero = 1e-5


# Misc. methods

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Actor(Module):

    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(Actor, self).__init__()

        self.model = Sequential(
            Linear(num_inputs, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, num_actions * 2)  # For each continuous action, a mu and a sigma
        )

    def forward(self, state: Tensor) -> (Tensor, Tensor):
        normal_params = self.model(state)

        mu_a_t = normal_params[:, 0]
        sigma_a_t = normal_params[:, 1]
        mu_p_t = normal_params[:, 2]
        sigma_p_t = normal_params[:, 3]

        # Guarantee that the standard deviation is not negative

        sigma_a_t = torch.exp(sigma_a_t) + zero
        sigma_p_t = torch.exp(sigma_p_t) + zero

        return mu_a_t, sigma_a_t, mu_p_t, sigma_p_t


class Critic(Module):

    def __init__(self, num_inputs, hidden_size=64, ):
        super(Critic, self).__init__()

        self.model = Sequential(
            Linear(num_inputs, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.model(state)


class Agent:

    def __init__(
            self, env: Env, gamma: float = 0.99, rollout_steps: int = 5, hidden_size: int = 64,
            actor_lr: float = 1e-4, critic_lr: float = 1e-4,
    ):

        # Parameter initialization

        self.env = env
        self.gamma = gamma
        self.rollout_steps = rollout_steps

        # Configure neural networks

        dim_obs = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        self.actor = Actor(num_inputs=dim_obs, num_actions=dim_action, hidden_size=hidden_size)
        self.critic = Critic(num_inputs=dim_obs, hidden_size=hidden_size)

        self.actor.optimizer = Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = Adam(params=self.critic.parameters(), lr=critic_lr)

        # Hooks into the models to collect gradients and topology

        wandb.watch(models=(self.actor, self.critic))

    def select_action(self, state: Tensor):

        mu_a_t, sigma_a_t, mu_p_t, sigma_p_t = self.actor(state)

        dist_a_t = Normal(loc=mu_a_t, scale=sigma_a_t)
        dist_p_t = Normal(loc=mu_p_t, scale=sigma_p_t)
        a_t = dist_a_t.sample()
        p_t = dist_p_t.sample()
        log_prob = dist_a_t.log_prob(a_t) + dist_p_t.log_prob(p_t)

        return (a_t, p_t), log_prob

    def rollout(self):

        states, rewards, log_probs = [], [], []

        # Get the initial state by resetting the environment

        state, _, _, _ = self.env.reset()

        for step in range(self.rollout_steps):
            # Start by appending the state to create the states trajectory

            states.append(state)

            # Perform action and pass to next state

            action, log_prob = self.select_action(Tensor(state))
            state, reward, _, _ = self.env.step(action=action)

            rewards.append(reward)
            log_probs.append(log_prob)

        return states, rewards, log_probs

    def train(self, training_steps: int = 1000):

        for step in range(training_steps):

            # Perform rollouts and sample trajectories

            states, rewards, log_probs = self.rollout()

            log_probs = torch.stack(log_probs, 0)
            value = [self.critic(state) for state in states]

            value = torch.stack(value, 0).squeeze()

            # Causality trick

            sum_rewards = []
            causal_reward = 0

            for reward in reversed(rewards):
                causal_reward = torch.clone(causal_reward + reward)
                sum_rewards.insert(0, causal_reward)

            sum_rewards = torch.stack(sum_rewards, 0)

            # Backpropagation to train Actor NN

            actor_loss = -torch.mean(torch.sum(log_probs * (sum_rewards - value.detach())))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Backpropagation to train Critic NN

            critic_loss = torch.mean((value - sum_rewards) ** 2)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            wandb.log({
                "rollout_avg_reward": torch.mean(sum_rewards),
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            })


if __name__ == '__main__':

    try:
        '''
            Define Microgrid parameters 
        '''

        exp_battery_params = BatteryParameters(
            soc_0=0.1,
            soc_max=0.9,
            soc_min=0.1,
            p_charge_max=0.5,
            p_discharge_max=0.5,
            efficiency=0.9,
            buy_price=0.6,
            sell_price=0.6,
            capacity=4
        )

        '''
            Define the simulation parameters
        '''

        n_participants = 10
        batch_size = 10
        agent_training_steps = 10
        agent_gamma = 0.99
        agent_rollout_steps = 24 * 30  # Hours * Days
        agent_actor_lr = 1e-3
        agent_critic_lr = 1e-3

        '''
            Setup all the configurations for Wandb
        '''

        wandb.init(
            project="p2p_price_rl_a2c",
            entity=os.environ.get("WANDB_ENTITY"),
            config={
                "n_participants": n_participants,
                "batch_size": batch_size,
                "training_steps": agent_training_steps,
                "gamma": agent_gamma,
                "rollout_steps": agent_rollout_steps,
                "agent_actor_lr": agent_actor_lr,
                "agent_critic_lr": agent_critic_lr,
            }
        )

        # Define the custom x-axis metric
        wandb.define_metric("test-step")

        # Define the x-axis for the plots: (avoids an issue with Wandb step autoincrement on each log call)

        wandb.define_metric("test_reward", step_metric='current_t')

        '''
            Run the simulator
        '''

        set_all_seeds(420)

        # Instantiate the environment

        mg_env = P2PA2C(n_participants=n_participants, battery_params=exp_battery_params, batch_size=batch_size)

        # Instantiate the agent
        agent = Agent(
            env=mg_env, gamma=agent_gamma, rollout_steps=agent_rollout_steps, critic_lr=agent_actor_lr,
            actor_lr=agent_actor_lr
        )

        # Launch the training

        agent.train(training_steps=agent_training_steps)

        # Finish the wandb process

        wandb.finish()

    except KeyboardInterrupt:
        wandb.finish()
