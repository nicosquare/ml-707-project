import wandb
import os

from dotenv import load_dotenv

from src.envs.p2p import P2PA2C

# Initialize Wandb for logging purposes

load_dotenv()
wandb.login(key=str(os.environ.get("WANDB_KEY")))

wandb.init(
    project="manual_p2p_price",
    entity=os.environ.get("WANDB_ENTITY")
)

# Instantiate the environment

mg_env = P2PA2C()

obs, reward, _, _ = mg_env.reset()
d_t, h_t, c_t, es_t, p_s = obs

while True:

    try:
        coeff_a_t = input('Type the coeff. a_t [0.2,1.2], to exit type "e":')
        coeff_p_t = input('Type the coeff. a_t [0.2,1.2], to exit type "e":')

        # Exit condition

        if coeff_a_t == 'e' or coeff_p_t == 'e':
            wandb.finish()
            break

        coeff_a_t = float(coeff_a_t)
        coeff_p_t = float(coeff_p_t)

        if 1.2 < coeff_a_t < 0.2 or 1.2 < float(coeff_p_t) < 0.2:
            raise ValueError

        # Perform the action

        obs, reward, _, _ = mg_env.step((coeff_a_t, coeff_p_t))
        d_t, h_t, c_t, es_t, p_s = obs

    except ValueError:
        print("Invalid value. Please input a value between 0.2 and 1")