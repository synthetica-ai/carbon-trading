import tensorflow as tf
from tensorflow import keras

# action 0 is left
# action 1 is right

# policy network
n_inputs = 4  # env.observation_space.shape[0] horizontal_pos, velocity, pole_angle, angular_velocity
model = keras.models.Sequential(
    [
        keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(1, activation="sigmoid"),  # outputs the proba of going left
    ]
)
