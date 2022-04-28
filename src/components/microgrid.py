import numpy as np

from torch import Tensor
from math import floor
from random import sample, random
from pymgrid import MicrogridGenerator as mg

from src.components.battery import Battery, BatteryParameters


class Microgrid:

    def __init__(
            self, n_participants: int, consumer_rate: float = 0.5, alpha: float = 0.333, beta: float = 0.333,
            k: float = 0.1, battery_params: BatteryParameters = None, batch_size: int = 1
    ):
        self._current_t = 0
        self.participants = []
        self.k = k
        self.beta = beta
        self.alpha = alpha

        # Configure the battery of the community

        self.battery = Battery(batch_size=batch_size, params=battery_params)

        # Randomly generate participants (as microgrids)

        env = mg.MicrogridGenerator(nb_microgrid=n_participants)
        env.generate_microgrid(verbose=True)

        self.participants = env.microgrids

        # Apply the consumer_rate configuration

        n_consumers = floor(n_participants * consumer_rate)

        for i in range(n_consumers):
            self.participants[i].architecture['PV'] = 0
            self.participants[i]._pv_ts *= 0

        for i in range(n_participants):
            self.participants[i]._pv_ts = ((self.participants[i]._pv_ts - self.participants[i]._pv_ts.min()) /
                                           self.participants[i]._pv_ts.max()) * 6
            self.participants[i]._load_ts = ((self.participants[i]._load_ts - self.participants[i]._load_ts.min()) /
                                             self.participants[i]._load_ts.max()) * 6

    def get_current_step_obs(self, coeff_a_t, coeff_p_t, size_of_slot: int = 24):
        """

        Get the states given a fixed time-slot

        :param size_of_slot: int
            Size in hours of a time-slot. TODO: Enable different time-slots sizes.
        :return: list
            List containing the measurements that form the state.
        """
        
        sur_sp = 0
        sur_batt = 0
        dem_sp = 0
        shortage_sp = 0
        dem_batt = 0
        shortage_batt = 0

        for participant in self.participants:

            participant_demand = participant._load_ts.iloc[self.get_current_step()][0]
            participant_generation = participant._pv_ts.iloc[self.get_current_step()][0]

            # Check surplus constraints

            if participant.architecture['PV'] == 1:  # if this is prosumer
                surplus = participant_generation - participant_demand
                if surplus > 0:
                    
                    if coeff_p_t < self.battery.sell_price:
                        
                        # We check how much of the surplus can be stored in the battery

                        p_charge, _, _ = self.battery.check_battery_constraints(input_power=Tensor(surplus))
                        sur_batt += p_charge
                        diff = surplus - p_charge.item()
                        sur_sp += diff
                    
                    else:

                        sur_sp += surplus

                else:

                    if coeff_a_t > self.battery.buy_price:

                        # We check how much of the shortage can be taken from the the battery

                        _, p_discharge, _ = self.battery.check_battery_constraints(input_power=Tensor(surplus))
                        shortage_batt += p_discharge
                        diff = p_discharge.item() - (-surplus)
                        shortage_sp += diff

                    else:

                        shortage_sp += (-surplus)
                    
            elif participant_generation == 0: # if this is consumer

                if coeff_a_t > self.battery.buy_price:

                    # We check how much of the shortage can be taken from the the battery

                    _, p_discharge, new_soc = self.battery.check_battery_constraints(input_power=Tensor(participant_demand))
                    dem_batt += p_discharge
                    diff = p_discharge.item() - participant_demand
                    dem_sp += diff

                else:

                    dem_sp += participant_demand

            # We might also do it as generation - demand

        # Compute the period of the day

        h_t = self.get_current_step() % size_of_slot + 1

        # Compute c_t: look at page 8 of the paper for better explanation

        d_h_t = dem_sp / len(self.participants)
        b_h = list(d_h_t * np.arange(0.25, 2, 0.25))
        b_h_t = sample(b_h, k=1)[0]  # return k-length list sampled from b_h
        alpha_t = 0.02

        c_t = alpha_t * dem_sp + b_h_t * dem_sp ** 2

        return self.battery.soc.item(), dem_sp, h_t, dem_batt, sur_sp, sur_batt, shortage_sp, shortage_batt, c_t

    def compute_current_step_cost(self, action: tuple):

        coeff_a_t, coeff_p_t = action
        new_soc, dem_sp, h_t, dem_batt, sur_sp, sur_batt, shortage_sp, shortage_batt, c_t = self.get_current_step_obs(coeff_a_t, coeff_p_t)

        # Compute costs

        consumer_cost_t = self.battery.buy_price*dem_batt + coeff_a_t*dem_sp
        prosumer_cost_t = self.battery.buy_price*shortage_batt + coeff_a_t*shortage_sp - self.battery.sell_price*sur_batt - coeff_p_t*sur_sp
        provider_cost_t = c_t + coeff_p_t*sur_sp - coeff_a_t*(dem_sp + shortage_sp)

        cost_t = (1 - self.alpha - self.beta) * provider_cost_t
        cost_t += self.alpha * consumer_cost_t
        cost_t += self.beta * prosumer_cost_t

        # Advance one step

        self._current_t += 1

        return cost_t, new_soc, (dem_sp + shortage_sp), h_t

    def get_current_step(self):
        """
            Returns the current time step. Allows running more than one year with the same data.
        Returns
        -------
            self.current_t: int
                Current microgrid time step
        """
        return self._current_t % 8760

    def reset_current_step(self):
        """
            Resets the current time step.
        Returns
        -------
            None
        """
        self._current_t = 0
