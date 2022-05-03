import gym
import torch
import numpy as np

from torch import Tensor
from gym import spaces

from src.components.battery import BatteryParameters
from src.components.microgrid import Microgrid
from src.utils.tensors import create_zeros_tensor, create_ones_tensor

inf = np.float64('inf')


class P2P(gym.Env):

    def __init__(self, n_participants: int = 10):
        self._microgrid = Microgrid(n_participants=n_participants)

        """
            Limits of observation space:
            soc => [0, 1]: State of Charge of the community battery.
            d_t => [0, 1]: Normalized sum of all the prosumers/consumers demand.
            h_t => [1, 24]: Period of the day
        """
        self.observation_space = spaces.Box(
            low=np.float32(np.array([0.0, 0.0, 1.0])),
            high=np.float32(np.array([1.0, 1.0, 24.0])),
            dtype=np.float32
        )

        """
            Coefficients of pricing equations
            
            coeff_a_t => [0.2, ... 1.0]: Possible coefficients
            coeff_p_t => [0.2, ... 1.0]: Possible coefficients
        
        """
        self.action_space = spaces.Discrete(25)

        # Define the tuples for the actions, we generate all the possible values

        base_list = np.arange(0.2, 1.2, 0.2)

        self.action_tuples = [(a, b) for a in base_list for b in base_list]

    def step(self, action):

        decoded_action = self.action_tuples[action]

        cost_t, next_state = self._microgrid.compute_current_step_cost(
            action=Tensor([decoded_action[0], decoded_action[1]])
        )

        # Normalize the new state

        next_state[:, 1] /= 60

        state = next_state
        reward = -cost_t
        done = self._microgrid.get_current_step() == 24 * 365
        info = {}

        if done:
            self.render()

        return state, reward, done, info

    def reset(self):
        self._microgrid.reset_current_step()
        return Tensor([0.1, 0, 1])

    def render(self, mode="human"):
        print('Render to be defined')


class P2PA2C(gym.Env):

    def __init__(self, n_participants: int = 10, battery_params: BatteryParameters = None, batch_size: int = 1):
        """
            Gym environment to simulate a P2P Microgrid scenario
        """

        self.batch_size = batch_size
        self.mg = Microgrid(n_participants=n_participants, battery_params=battery_params, batch_size=batch_size)

        """
            Limits of observation space:
            soc => [0, 1]: State of Charge of the community battery.
            d_t => [0, 1]: Normalized sum of all the prosumers/consumers demand.
            h_t => [1, 24]: Period of the day
        """
        self.observation_space = spaces.Box(
            low=np.float32(np.array([0.0, 0.0, 1.0])),
            high=np.float32(np.array([1, 1, 24.0])),
            dtype=np.float32
        )

        """
            Coefficients of pricing equations

            coeff_a_t => [0.2,1.2]
            coeff_p_t => [0.2,1.2]

        """
        self.action_space = spaces.Box(
            low=0.2,
            high=1.2,
            shape=(2,),
            dtype=np.float32
        )

    def step(self, action: tuple):
        cost_t, next_state = self.mg.compute_current_step_cost(
            action=action
        )

        # Normalize the new state

        next_state[:, 1] /= 60

        state = next_state
        reward = -cost_t
        done = False
        info = {}

        return state, reward, done, info

    def reset(self):
        # Resetting the microgrid
        self.mg.reset_current_step()
        # Building the initial state
        initial_state = torch.stack((
            self.mg.battery.initialize_soc(),
            create_zeros_tensor(size=self.batch_size),
            create_ones_tensor(size=self.batch_size)
        ), dim=1)

        return initial_state, 0, False, {}

    def render(self, mode="human"):
        print('Render to be defined')
