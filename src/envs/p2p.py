import gym
import numpy as np
from gym import spaces

from src.components.microgrid import Microgrid

infinity = np.float('inf')


class P2P(gym.Env):

    def __init__(self, n_participants: int = 10):
        self._microgrid = Microgrid(n_participants=n_participants)

        """
            Limits of observation space:
            d_t => [0, inf]: Sum of all the prosumers/consumers demand
            h_t => [1, 24]: Period of the day
            c_t => [0, inf]: Service provider cost
            es_t => [0, inf]: Sum of the available surplus of energy
        """
        self.observation_space = spaces.Box(
            low=np.float32(np.array([0.0, 1.0, 0.0, 0.0])),
            high=np.float32(np.array([infinity, 24.0, infinity, infinity])),
            dtype=np.float32
        )

        """
            Coefficients of pricing equations
            
            coeff_a_t => [0.2, ... 1.0]: Possible coefficients
            coeff_p_t => [0.2, ... 1.0]: Possible coefficients
        
        """
        self.action_space = spaces.Discrete(25)

        # Define the tuples for the actions, we generate all the possible values

        self.action_tuples = []
        base_list = np.arange(0.2, 1.2, 0.2)

        for i in base_list:
            for j in base_list:
                self.action_tuples.append((i, j))

    def _observe(self):

        d_t, h_t, c_t, es_t, _, _, _ = self._microgrid.get_current_step_obs()

        return d_t, h_t, c_t, es_t

    def step(self, action):

        cost_t, d_t_next, h_t_next, c_t_next, es_t_next = self._microgrid.compute_current_step_cost(
            action=self.action_tuples[action]
        )

        state = d_t_next, h_t_next, c_t_next, es_t_next
        reward = -cost_t
        done = self._microgrid.get_current_step() == 24 * 365
        info = {}

        if done:
            self.render()

        return state, reward, done, info

    def reset(self):
        self._microgrid.reset_microgrid()
        return self._observe()

    def render(self, mode="human"):
        self._microgrid.plot_all()