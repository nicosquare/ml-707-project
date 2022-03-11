import gym
import numpy as np
from gym import spaces
from pymgrid import MicrogridGenerator as mg

infinity = np.float('inf')


class P2PEnv(gym.Env):

    def __init__(self):
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
        self.action_space = spaces.MultiDiscrete([5, 5])

        """
        
             TODO:
             
             - Will this be a generic environment definition
             - Use pymgrid to define a number of microgrids
             - Extract data from pymgrid
             - Create data for es_t
             - Create data for h_t 
             - Create data for c_t 
        
        """
