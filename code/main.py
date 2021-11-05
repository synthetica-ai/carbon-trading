import tensorflow as tf

# from termcolor import colored
import gym

from env.env import CarbonEnv

# from data.data_functions import train_input_fn, eval_input_fn
# from utils.util_functions import create_meta_parameters
# from models.models import model_skeleton
from training.training_functions import PolicyGradient


env = CarbonEnv()
model = PolicyGradient(env)
model.train()
