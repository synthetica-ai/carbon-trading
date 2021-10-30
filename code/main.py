import tensorflow as tf
from termcolor import colored


# from env.enviroments import SimosFoodGroup
from data.data_functions import  train_input_fn
from utils.util_functions import create_meta_parameters
from models.models import model_skeleton
from training.training_functions import train



# Metaparameters
metaparameters = create_meta_parameters()

# Data sources
validation_dataset = eval_input_fn(metaparameters)

# Train operation
model = train(validation_dataset, metaparameters,config)

