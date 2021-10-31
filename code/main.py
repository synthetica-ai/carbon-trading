import tensorflow as tf
from termcolor import colored
import gym

# from env.enviroments import SimosFoodGroup
from data.data_functions import  train_input_fn, eval_input_fn
from utils.util_functions import create_meta_parameters
from models.models import model_skeleton
from training.training_functions import train, PolicyGradient



env = gym.make("CartPole-v0")
model = PolicyGradient(env)
model.train()
