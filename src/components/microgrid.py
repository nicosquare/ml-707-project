import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import floor
from random import sample, random
from pymgrid import MicrogridGenerator as mg


class Microgrid:

    def __init__(
            self, n_participants: int, consumer_rate: float = 0.5, alpha: float = 0.7, beta: float = 0.1,
            k: float = 0.1, n_steps_avg: int = 24 * 30
        ):

        self._current_t = 0
        self.participants = []
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.n_steps_avg = n_steps_avg

        # Configure accumulator variables

        self.avg_consumer_cost, self.avg_prosumer_cost, self.avg_provider_cost, self.avg_operation_cost = 0, 0, 0, 0

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

        # Configure DataFrames for visualization

        self.df_operation_cost = pd.DataFrame(columns=['operation_cost'])
        self.df_sp_cost = pd.DataFrame(columns=['sp_cost'])
        self.df_prosumers_cost = pd.DataFrame(columns=['prosumers_cost'])
        self.df_consumers_cost = pd.DataFrame(columns=['consumers_cost'])
        self.df_coeff_a_t = pd.DataFrame(columns=['coeff_a_t'])
        self.df_coeff_p_t = pd.DataFrame(columns=['coeff_p_t'])

    def get_current_step_obs(self, size_of_slot: int = 24):
        """

        Get the states given a fixed time-slot

        :param size_of_slot: int
            Size in hours of a time-slot. TODO: Enable different time-slots sizes.
        :return: list
            List containing the measurements that form the state.
        """
        sum_e_t = 0  # total consumed energy that sp buys from UG
        d_h = []  # this is needed to compute c/p costs
        prosumers_surplus = []
        prosumers_shortage = []

        for participant in self.participants:
            participant_consumption = participant._load_ts.iloc[self.get_current_step()][0]
            participant_generation = participant._pv_ts.iloc[self.get_current_step()][0]

            # Demand is sometimes more than consumption or less than consumption

            demand_variation = 0.05 * participant_consumption if random() < 0.5 else -0.05 * participant_consumption
            participant_demand = participant_consumption + demand_variation

            d_h.append(participant_demand)

            # Check surplus constraints

            if participant.architecture['PV'] == 1:  # if this is prosumer
                surplus = participant_generation - participant_consumption
                if surplus > 0:
                    prosumers_surplus.append(surplus)
                    # es_t += surplus
                else:
                    prosumers_shortage.append(-surplus)
                    sum_e_t += -surplus
            elif participant_generation == 0:
                sum_e_t += participant_consumption
            # We might also do it as generation - demand

        # Compute the period of the day

        h_t = self.get_current_step() % size_of_slot + 1

        # Compute c_t: look at page 8 of the paper for better explanation

        d_h_t = np.mean(d_h)
        b_h = list(d_h_t * np.arange(0.25, 2, 0.25))
        b_h_t = sample(b_h, k=1)[0]  # return k-length list sampled from b_h
        alpha_t = 0.02

        c_t = alpha_t * sum_e_t + b_h_t * sum_e_t ** 2

        return np.sum(d_h), h_t, c_t, np.sum(prosumers_surplus), np.sum(
            prosumers_shortage), d_h, prosumers_surplus, prosumers_shortage

    def compute_current_step_cost(self, action: tuple, logging: bool = True):

        coeff_a_t, coeff_p_t = action
        d_t, h_t, c_t, es_t, p_s, d_h, prosumers_surplus, prosumers_shortage = self.get_current_step_obs()

        consumer_cost_t, prosumer_cost_t = self.compute_consumer_prosumer_cost(
            coeff_a_t=coeff_a_t, coeff_p_t=coeff_p_t, demand_list=d_h, prosumers_surplus=prosumers_surplus,
            prosumers_shortage=prosumers_shortage
        )
        provider_cost_t = self.service_provider_cost(
            c_t=c_t, coeff_a_t=coeff_a_t, coeff_p_t=coeff_p_t, prosumers_surplus=prosumers_surplus,
            prosumers_shortage=prosumers_shortage
        )

        cost_t = (1 - self.alpha - self.beta) * provider_cost_t
        cost_t += self.alpha * consumer_cost_t
        cost_t += self.beta * prosumer_cost_t

        # Accumulate the values for averaging

        self.avg_consumer_cost += consumer_cost_t
        self.avg_prosumer_cost += prosumer_cost_t
        self.avg_provider_cost += provider_cost_t
        self.avg_operation_cost += cost_t

        # Store data in DataFrames

        self.df_consumers_cost.loc[len(self.df_consumers_cost)] = consumer_cost_t
        self.df_prosumers_cost.loc[len(self.df_prosumers_cost)] = prosumer_cost_t
        self.df_sp_cost.loc[len(self.df_sp_cost)] = provider_cost_t
        self.df_operation_cost.loc[len(self.df_operation_cost)] = cost_t
        self.df_coeff_a_t.loc[len(self.df_coeff_a_t)] = coeff_a_t
        self.df_coeff_p_t.loc[len(self.df_coeff_p_t)] = coeff_p_t

        wandb.log({
            "consumer_cost": consumer_cost_t,
            "prosumer_cost": prosumer_cost_t,
            "provider_cost": provider_cost_t,
            "operation_cost": cost_t,
            "coeff_a_t": coeff_a_t,
            "coeff_p_t": coeff_p_t,
            "utility_cost": c_t,
        })

        if logging:

            log_dict = {
                "current_t": self._current_t,
                "consumer_cost": consumer_cost_t,
                "prosumer_cost": prosumer_cost_t,
                "provider_cost": provider_cost_t,
                "operation_cost": cost_t,
                "coeff_a_t": coeff_a_t,
                "coeff_p_t": coeff_p_t,
                "utility_cost": c_t,
            }

        #     if self._current_t % self.n_steps_avg == 0:

        #         # Average the costs according to the defined n_steps_avg

        #         self.avg_consumer_cost /= self.n_steps_avg
        #         self.avg_prosumer_cost /= self.n_steps_avg
        #         self.avg_provider_cost /= self.n_steps_avg
        #         self.avg_operation_cost /= self.n_steps_avg

        #         # Add averages to the log dictionary

        #         log_dict["avg_consumer_cost"] = self.avg_consumer_cost
        #         log_dict["avg_prosumer_cost"] = self.avg_prosumer_cost
        #         log_dict["avg_provider_cost"] = self.avg_provider_cost
        #         log_dict["avg_operation_cost"] = self.avg_operation_cost

        #         # Reset the averages

        #         self.avg_consumer_cost = 0
        #         self.avg_prosumer_cost = 0
        #         self.avg_provider_cost = 0
        #         self.avg_operation_cost = 0

            wandb.log(log_dict)

        # Advance one step

        self._current_t += 1

        d_t_next, h_t_next, c_t_next, es_t_next, p_s_next, _, _, _ = self.get_current_step_obs()

        return cost_t, d_t_next, h_t_next, c_t_next, es_t_next, p_s_next

    def compute_consumer_prosumer_cost(self, coeff_a_t: float, coeff_p_t: float, demand_list: list,
                                       prosumers_surplus: list, prosumers_shortage: list):

        total_consumer_cost = 0
        total_prosumer_cost = 0

        for participant_ix in range(len(self.participants)):

            participant_consumption = self.participants[participant_ix]._load_ts.iloc[self.get_current_step()][0]

            if self.participants[participant_ix].architecture['PV'] == 0:  # if this is consumer

                a_t = coeff_a_t * participant_consumption

                u_t = self.k * (demand_list[participant_ix] - participant_consumption) ** 2

                # Total cost of all consumers in the defined currency

                total_consumer_cost += u_t + a_t

            else:
                u_t = self.k * (demand_list[participant_ix] - participant_consumption) ** 2
                total_prosumer_cost += u_t

            participant_ix += 1

        total_prosumer_cost += coeff_a_t * np.sum(prosumers_shortage)
        total_prosumer_cost -= coeff_p_t * np.sum(prosumers_surplus)

        return total_consumer_cost, total_prosumer_cost

    def service_provider_cost(self, c_t: float, coeff_a_t: float, coeff_p_t: float, prosumers_surplus: list,
                              prosumers_shortage: list):

        sum_a_t = 0
        sum_p_t = 0

        # participant_ix = 0

        for participant in self.participants:
            participant_generation = participant._pv_ts.iloc[self.get_current_step()][0]
            if participant_generation == 0:
                participant_consumption = participant._load_ts.iloc[self.get_current_step()][0]
                sum_a_t += coeff_a_t * participant_consumption

        sum_p_t += coeff_p_t * np.sum(prosumers_surplus)
        sum_a_t += coeff_a_t * np.sum(prosumers_shortage)

        return c_t + sum_p_t - sum_a_t

    def get_current_step(self):
        """
            Returns the current time step. Allows running more than one year with the same data.
        Returns
        -------
            self.current_t: int
                Current microgrid time step
        """
        return self._current_t % 8760

    def set_current_step(self, time_step: int):
        """
            Set the time step to a certain point
        Parameters
        ----------
        time_step: int
            Value of the time step.
        """
        self._current_t = time_step

    def reset_current_step(self):
        """
            Resets the current time step.
        Returns
        -------
            None
        """
        self._current_t = 0

    def plot_sp_cost(self):
        self.df_sp_cost.plot()
        plt.show()

    def plot_consumers_cost(self):
        self.df_consumers_cost.plot()
        plt.show()

    def plot_prosumers_cost(self):
        self.df_prosumers_cost.plot()
        plt.show()

    def plot_operation_cost(self):
        self.df_operation_cost.plot()
        plt.show()

    def plot_coeff_a_t(self):
        self.df_coeff_a_t.plot()
        plt.show()

    def plot_coeff_p_t(self):
        self.df_coeff_p_t.plot()
        plt.show()

    def save_all_csv(self):
        self.merged= pd.concat([self.df_sp_cost, self.df_consumers_cost, self.df_prosumers_cost, self.df_operation_cost, 
                            self.df_coeff_a_t, self.df_coeff_p_t], axis=1)
        self.merged.to_csv("/home/shahad.hardan/Documents/ML707/ml-707-project/notebooks/merge_7_1_2.csv")
        

    def plot_all(self):
        self.plot_sp_cost()
        self.plot_consumers_cost()
        self.plot_prosumers_cost()
        self.plot_operation_cost()
        self.plot_coeff_a_t()
        self.plot_coeff_p_t()
    
    def save_all(self): 
        self.save_all_csv()
        