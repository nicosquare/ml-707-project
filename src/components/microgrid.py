import numpy as np
import torch

from torch import Tensor
from math import floor
from pymgrid import MicrogridGenerator as mg

from src.components.battery import Battery, BatteryParameters
from src.utils.tensors import create_zeros_tensor, create_ones_tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)


class Microgrid:

    def __init__(
        self, n_participants: int, consumer_rate: float = 0.5, alpha: float = 0.333, beta: float = 0.333,
        k: float = 0.1, battery_params: BatteryParameters = None, batch_size: int = 1, scaling_multiplier: int = 6,
        coeff_c_t: float = 0.2
    ):
        self._current_t = 0
        self.n_participants = n_participants
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.batch_size = batch_size
        self.scaling_multiplier = scaling_multiplier
        self.coeff_c_t = coeff_c_t

        # Configure the battery of the community

        self.battery = Battery(batch_size=batch_size, params=battery_params)

        # Initialize participants data

        self.pv = None
        self.load = None
        self.is_prosumer = Tensor([True] * self.n_participants)

        self.initialize_data(n_consumers=floor(self.n_participants * consumer_rate))

    def initialize_data(self, n_consumers: int):
        """

            Get information from pymgrid and configure all the participants data according to the case parameters.

        :param n_consumers: Number of participants who are just consumers
        :return:
            None
        """
        # Randomly generate participants (as microgrids)

        env = mg.MicrogridGenerator(nb_microgrid=self.n_participants)
        env.generate_microgrid(verbose=True)

        # Extract the information from pymgrid

        participants = env.microgrids

        self.load = Tensor(np.array([participant._load_ts.fillna(0).to_numpy() for participant in participants]))
        self.pv = Tensor(np.array([participant._pv_ts.fillna(0).to_numpy() for participant in participants]))

        # Scale the load and pv to maintain

        for i in range(self.n_participants):

            self.load[i] = ((self.load[i] - self.load[i].min()) / self.load[i].max()) * self.scaling_multiplier
            self.pv[i] = ((self.pv[i] - self.pv[i].min()) / self.pv[i].max()) * self.scaling_multiplier

        # Apply the consumer_rate configuration

        self.pv[0:n_consumers] = 0
        self.is_prosumer[0:n_consumers] = False

    def get_current_step_obs(self, coeff_a_t, coeff_p_t, size_of_slot: int = 24):
        """

        Get the states given a fixed time-slot

        :param coeff_p_t:
        :param coeff_a_t:
        :param size_of_slot: int
            Size in hours of a time-slot. TODO: Enable different time-slots sizes.
        :return: list
            List containing the measurements that form the state.
        """

        sur_sp = create_zeros_tensor(size=self.batch_size)
        sur_batt = create_zeros_tensor(size=self.batch_size)
        dem_sp = create_zeros_tensor(size=self.batch_size)
        shortage_sp = create_zeros_tensor(size=self.batch_size)
        dem_batt = create_zeros_tensor(size=self.batch_size)
        shortage_batt = create_zeros_tensor(size=self.batch_size)

        for p_ix in range(self.n_participants):

            # Get the participant step values

            participant_demand = self.load[:, self.get_current_step()][p_ix]
            participant_demand = Tensor([participant_demand]).to(device)
            participant_generation = self.pv[:, self.get_current_step()][p_ix]
            participant_generation = Tensor([participant_generation]).to(device)

            if not self.is_prosumer[p_ix]:  # Is consumer

                # Get energy from the battery first if the prize is suitable

                p_discharge = torch.where(
                    coeff_a_t > self.battery.buy_price,
                    self.battery.check_battery_constraints(input_power=participant_demand)[1],
                    create_zeros_tensor(size=self.batch_size)
                )

                # Compute the new SoC

                self.battery.compute_new_soc(
                    p_charge=create_zeros_tensor(size=self.batch_size),
                    p_discharge=p_discharge
                )

                # Update the accumulator tensors

                dem_batt += p_discharge
                diff = p_discharge - participant_demand
                dem_sp += torch.abs(diff)

            else:  # Is prosumer

                surplus = participant_generation - participant_demand

                if surplus > 0:

                    # We check how much of the surplus can be stored in the battery if the price is suitable

                    p_charge = torch.where(
                        coeff_p_t < self.battery.sell_price,
                        self.battery.check_battery_constraints(input_power=participant_demand)[0],
                        create_zeros_tensor(size=self.batch_size)
                    )

                    # Compute the new SoC

                    self.battery.compute_new_soc(
                        p_charge=p_charge,
                        p_discharge=create_zeros_tensor(size=self.batch_size)
                    )

                    # Update the accumulator tensors

                    sur_batt += p_charge
                    diff = surplus - p_charge
                    sur_sp += diff

                else:  # There is shortage

                    # We check how much of the shortage can be taken from the battery is the price is suitable

                    p_discharge = torch.where(
                        coeff_a_t > self.battery.buy_price,
                        self.battery.check_battery_constraints(input_power=participant_demand)[1],
                        create_zeros_tensor(size=self.batch_size)
                    )

                    # Compute the new SoC

                    self.battery.compute_new_soc(
                        p_charge=create_zeros_tensor(size=self.batch_size),
                        p_discharge=p_discharge
                    )

                    # Update the accumulator tensors

                    shortage_batt += p_discharge
                    diff = p_discharge - (-surplus)
                    shortage_sp += torch.abs(diff)

        # Compute the period of the day

        h_t = self.get_current_step() % size_of_slot + 1

        # Compute c_t

        c_t = self.coeff_c_t * dem_sp

        return self.battery.soc, dem_sp, h_t, dem_batt, sur_sp, sur_batt, shortage_sp, shortage_batt, c_t

    def compute_current_step_cost(self, action: tuple):

        coeff_a_t, coeff_p_t = action
        new_soc, dem_sp, h_t, dem_batt, sur_sp, sur_batt, shortage_sp, shortage_batt, c_t = self \
            .get_current_step_obs(coeff_a_t, coeff_p_t)

        # Compute costs

        consumer_cost_t = self.battery.buy_price * dem_batt + coeff_a_t * dem_sp  # Consumer
        prosumer_cost_t = self.battery.buy_price * shortage_batt + coeff_a_t * shortage_sp  # Prosumer
        prosumer_cost_t -= self.battery.sell_price * sur_batt - coeff_p_t * sur_sp
        provider_cost_t = c_t + coeff_p_t * sur_sp - coeff_a_t * (dem_sp + shortage_sp)  # Service provider

        cost_t = (1 - self.alpha - self.beta) * provider_cost_t
        cost_t += self.alpha * consumer_cost_t
        cost_t += self.beta * prosumer_cost_t

        # Build next state

        next_state = torch.stack((
            new_soc,
            (dem_sp + shortage_sp),
            create_ones_tensor(size=self.batch_size) * h_t
        ), dim=1)

        # Advance one step

        self._current_t += 1

        return cost_t, next_state

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
