import gym
import numpy as np
from gym import spaces

from src.components.microgrid import Microgrid

inf = np.float64('inf')


class P2P(gym.Env):

    def __init__(self, n_participants: int = 10):
        self._microgrid = Microgrid(n_participants=n_participants)

        """
            Limits of observation space:
            d_t => [0, inf]: Sum of all the prosumers/consumers demand
            h_t => [1, 24]: Period of the day
            c_t => [0, inf]: Service provider cost
            es_t => [0, inf]: Sum of the available surplus of energy
            p_s => [0, inf]: Sum of the prosumers' shortage
        """
        self.observation_space = spaces.Box(
            low=np.float32(np.array([0.0, 1.0, 0.0, 0.0, 0.0])),
            high=np.float32(np.array([inf, 24.0, inf, inf, inf])),
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

        # for i in base_list:
        #     for j in base_list:
        #         self.action_tuples.append((i, j))

    def _observe(self):
        d_t, h_t, c_t, es_t, p_s, _, _, _ = self._microgrid.get_current_step_obs()

        return d_t, h_t, c_t, es_t, p_s

    def step(self, action):
        cost_t, d_t_next, h_t_next, c_t_next, es_t_next, p_s_next = self._microgrid.compute_current_step_cost(
            action=self.action_tuples[action]
        )

        state = d_t_next, h_t_next, c_t_next, es_t_next, p_s_next
        reward = -cost_t
        done = self._microgrid.get_current_step() == 24 * 365-1
        info = {}

        if done:
            self.render()

        return state, reward, done, info

    def reset(self):
        self._microgrid.reset_current_step()
        return self._observe()

    def render(self, mode="human"):
        #self._microgrid.plot_all()
        pass
        # self._microgrid.plot_all()


# class P2PA2C(gym.Env):

#     def __init__(self, n_participants: int = 10, logging: bool = True):
#         """
#             Gym environment to simulate a P2P Microgrid scenario
#         """

#         self.logging = logging
#         self._microgrid = Microgrid(n_participants=n_participants)

#         """
#             Limits of observation space:
#             d_t => [0, inf]: Sum of all the prosumers/consumers demand
#             h_t => [1, 24]: Period of the day
#             c_t => [0, inf]: Service provider cost
#             es_t => [0, inf]: Sum of the available surplus of energy
#             p_s => [0, inf]: Sum of the prosumers' shortage
#         """
#         self.observation_space = spaces.Box(
#             low=np.float32(np.array([0.0, 1.0, 0.0, 0.0, 0.0])),
#             high=np.float32(np.array([inf, 24.0, inf, inf, inf])),
#             dtype=np.float32
#         )

#         """
#             Coefficients of pricing equations

#             coeff_a_t_mean => [0.2,1.2]: at mean
#             coeff_a_t_stdd => [0.2,1.2]: at standard deviation
#             coeff_p_t_mean => [0.2,1.2]: pt mean
#             coeff_p_t_stdd => [0.2,1.2]: pt standard deviation

#         """
#         self.action_space = spaces.Box(
#             low=0.2,
#             high=1.2,
#             shape=(4,),
#             dtype=np.float32
#         )

#     def _observe(self):
#         d_t, h_t, c_t, es_t, p_s, _, _, _ = self._microgrid.get_current_step_obs()

#         return d_t, h_t, c_t, es_t, p_s

#     def step(self, action: tuple):
#         cost_t, d_t_next, h_t_next, c_t_next, es_t_next, p_s_next = self._microgrid.compute_current_step_cost(
#             action=action, logging=self.logging
#         )

#         state = d_t_next, h_t_next, c_t_next, es_t_next, p_s_next
#         reward = -cost_t
#         done = False
#         info = {}

#         if done:
#             self.render()

#         return state, reward, done, info

#     def reset(self):
#         self._microgrid.reset_current_step()
#         return self._observe(), 0, False, {}

#     def render(self, mode="human"):
#         self._microgrid.plot_all()

#     def set_logging(self, enabled: bool):
#         self.logging = enabled

#     def restore(self, time_step: int):
#         self._microgrid.set_current_step(time_step=time_step)
