import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import empty_eager_fallback
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_probability as tfp

import numpy as np
from env.env import CarbonEnv
import numpy as np

# from scipy.stats import ttest_rel
# from tqdm import tqdm


# from data.data_functions import baseline_input_fn


class CarbonModel(tf.keras.Model):
    """
    `CarbonModel` class used in PolicyNet and BaselineNet.


    Takes inputs:
    * `state_dict` = {"contracts_state":`contracts_tensor`,
                    "ships_state":`ships_tensor`,
                    "contracts_mask":`contracts_mask`,
                    "ships_mask":`ships_mask`}

    Outputs:
    * `logits`

    """  # TODO bug is here

    def __init__(self, output_size, embedding_size, policynet_flag):
        super(CarbonModel, self).__init__()
        """
        `__init__` is executed when the instance is first initiated
        """
        # Initializing instance parameters

        self.output_size = output_size
        self.embedding_size = embedding_size
        self.policynet_flag = policynet_flag
        # self.contracts_input_shape = contracts_input_shape
        # self.fleet_input_shape = fleet_input_shape

        # self.contracts_input_layer = tf.keras.Input(shape=self.contracts_input_shape,batch_size=self.batch_size)
        # self.fleet_input_layer = tf.keras.Input(shape=self.fleet_input_shape,batch_size=self.batch_size)

        self.dense1 = tf.keras.layers.Dense(256, activation="relu", name="dense_layer_1")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu", name="dense_layer_2")
        self.dense3 = tf.keras.layers.Dense(128, activation="relu", name="dense_layer_3")
        self.dense4 = tf.keras.layers.Dense(64, activation="relu", name="dense_layer_4")
        self.embedding_layer_1 = tf.keras.layers.Dense(
            self.embedding_size, activation="relu", name="embdedding_layer_1"
        )
        self.embedding_layer_2 = tf.keras.layers.Dense(
            self.embedding_size, activation="relu", name="embdedding_layer_2"
        )
        self.context_layer = tf.keras.layers.Dense(64, activation="relu", name="context_layer")
        self.output_layer = tf.keras.layers.Dense(self.output_size, activation="linear", name="output_layer")

    def call(self, state_dict):
        """
        Gets executed when a CarbonModel instance is called.
        Inputs:
        * `state_dict`: {"contracts_state":contracts_tensor,
                          "ships_state":ships_tensor,
                          "contracts_mask":contracts_mask,
                          "ships_mask":ships_mask}

        """

        self.contracts_tensor = state_dict["contracts_state"]
        self.ships_tensor = state_dict["ships_state"]
        self.contracts_mask = state_dict["contracts_mask"]
        self.ships_mask = state_dict["ships_mask"]

        x = self.dense1(self.contracts_tensor)
        x = self.dense2(x)
        contracts_embedding = self.embedding_layer_1(x)

        y = self.dense3(self.ships_tensor)
        y = self.dense4(y)
        fleet_embedding = self.embedding_layer_2(y)

        # gia poio ship milaw
        # current_agent =

        context = tf.concat([contracts_embedding, fleet_embedding], 1)
        context = self.context_layer(context)

        # reduce the context to get a 13,1 logits vector instead of 4,13
        context = tf.math.reduce_mean(context, axis=0)
        logits = self.output_layer(context)

        if self.policynet_flag:

            # an exw dialeksh to policynet bale maska
            # opou h maska twn contracts einai 0 bale -np.Infinity
            # h actions_boolean_mask exei dimension
            contracts_bm = tf.equal(self.contracts_mask, 0)
            actions_bm = tf.where(contracts_bm, tf.repeat(tf.constant(False), 3), tf.repeat(tf.constant(True), 3))
            actions_bm = tf.reshape(actions_bm, [-1])
            actions_bm = tf.expand_dims(actions_bm, axis=1)
            # bazw ena teleutaio true gia to 13 action tou select nothing
            actions_bm = tf.concat((actions_bm, tf.constant(True, shape=(1, 1))), axis=0)
            logits = tf.where(actions_bm, float("-inf"), logits)

        return logits


# Code version 1

# Baseline or ValueNet
class BaselineNet(object):
    """
    `BasileNet` takes a state as input and outputs the value of that state.
    The value of the state is actually the expected return of the input state
    """

    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.baseline_model = CarbonModel(embedding_size=self.embedding_size, output_size=1, policynet_flag=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)

    def forward(self, state_dict):
        output = tf.squeeze(self.baseline_model(state_dict))
        return output

    def update(self, state_dict, target):
        """
        A good idea for the loss of the value function is the monte carlo error
        loss_value = sum_over_samples[estimated_advantage^2]
        """
        with tf.GradientTape() as tape:
            predictions = self.forward(state_dict)
            loss = tf.keras.losses.mean_squared_error(y_true=target, y_pred=predictions)
        grads = tape.gradient(loss, self.baseline_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.baseline_model.trainable_weights))


# PolicyNet
class PolicyNet(object):
    """
    `PolicyNet` is used to learn the policy.
    It takes a state as input and outputs an action for that state

    """

    def __init__(self, embedding_size, output_size):
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.policy_model = CarbonModel(
            embedding_size=self.embedding_size, output_size=self.output_size, policynet_flag=True
        )

    def action_distribution(self, state_dict):
        logits = self.policy_model(state_dict)
        return logits, tfp.distributions.Categorical(logits=logits)

    def sample_action(self, state_dict):
        sampled_actions = self.action_distribution(state_dict).sample().numpy()
        return sampled_actions

