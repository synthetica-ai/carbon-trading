import tensorflow as tf
from termcolor import colored


# from env.enviroments import SimosFoodGroup
from data.data_functions import  train_input_fn, eval_input_fn
from utils.util_functions import create_meta_parameters
from models.models import model_skeleton
from training.training_functions import train



# env = gym.make("CartPole-v0")
model = PolicyGradient(env)
model.train()
