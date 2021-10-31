import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import Input, Model
import tensorflow_probability as tfp
import numpy as np


import numpy as np
from scipy.stats import ttest_rel
import tensorflow as tf
from tqdm import tqdm


# from data.data_functions import baseline_input_fn
# from env.enviroments import SimosFoodGroup
from models.layers import GraphAttentionEncoder,GraphAttentionDecoder

# Code version 1 

# Baseline Model
class BaselineNet():
    def __init__(self, input_size, output_size):
        self.model = keras.Sequential(
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(64, activation="relu", name="relu_layer"),
                layers.Dense(output_size, activation="linear", name="linear_layer")
            ],
            name="baseline")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)

    def forward(self, observations):
        output = tf.squeeze(self.model(observations))
        return output

    def update(self, observations, target):
        with tf.GradientTape() as tape:
            predictions = self.forward(observations)
            loss = tf.keras.losses.mean_squared_error(y_true=target, y_pred=predictions)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))





class PolicyNet():
    def __init__(self, input_size, output_size):

        contract_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="contracts")
        fleet_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="fleet")
        
        # Encoding part
        initializer = tf.keras.initializers.GlorotUniform()
        x = layers.Dense(256, activation="relu", kernel_initializer=initializer, name="enc_dense_1")(contracts_input)
        x = layers.Dense(64, activation="relu", kernel_initializer=initializer, name="enc_dense_2")(x)
        embedding = layers.Dense(32, activation="relu", kernel_initializer=initializer, name="enc_dense_3")(x)

        # Contracts head
        #TODO concat fleet with embedding
        x = layers.Dense(32, activation="relu", kernel_initializer=initializer,name="contract_dense_1")(embedding)
        x = layers.Dense(64, activation="relu", kernel_initializer=initializer, name="contract_dense_2")(x)
        x = layers.Dense(256, activation="relu", kernel_initializer=initializer, name="contract_dense_3")(x)
        contracts_output = layers.Dense(output_size, activation="relu", kernel_initializer=initializer, name="contract_output")(x)

        #TODO concat fleet with embedding
        x = layers.Dense(32, activation="relu", kernel_initializer=initializer,name="speed_dense_1")(embedding)
        x = layers.Dense(64, activation="relu", kernel_initializer=initializer, name="speed_dense_2")(x)
        x = layers.Dense(256, activation="relu", kernel_initializer=initializer, name="speed_dense_3")(x)
        speed_output = layers.Dense(output_size, activation="relu", kernel_initializer=initializer, name="speed_output")(x)

        model = keras.Model(inputs=[contract_input, fleet_input], 
              outputs =[contract_output, speed_output], 
              name="CarbonNetPolicy")

        self.model = keras.Sequential(
            layers=[
                keras.Input(shape=(input_size,)),
                layers.Dense(64, activation="relu", name="relu_layer"),
                layers.Dense(output_size, activation="linear", name="linear_layer")
            ],
            name="policy")

    def action_distribution(self, observations):
        logits = self.model(observations)
        return tfp.distributions.Categorical(logits=logits)

    def sample_action(self, observations):
        sampled_actions = self.action_distribution(observations).sample().numpy()
        return sampled_actions


# Code version 2

# def set_decode_type(model, decode_type):
#     model.set_decode_type(decode_type)
#     model.decoder.set_decode_type(decode_type)

# class AttentionModel(tf.keras.Model):

#     def __init__(self,params):

#         super().__init__()

#         self.embedding_dim = params['embedding_dim']
#         self.n_encode_layers = params['n_encode_layers']
#         self.decode_type = None
#         self.problem = SimosFoodGroup
#         self.n_heads = params['n_heads']
#         self.feed_forward_hidden = params['feed_forward_hidden']
#         self.tanh_clipping = params['tanh_clipping']
#         self.capacities = params['capacities']
#         self.distance_tensor = tf.constant(params['distance_tensor'],dtype = tf.float32)
#         self.duration_tensor = tf.constant(params['duration_tensor'],dtype = tf.float32)
#         self.truck_tensor = tf.sort(tf.constant(params['truck_capacities']), direction ='DESCENDING')
        
#         self.start_time_windows = tf.reshape(tf.concat([tf.constant([[0.0]]),params['start_windows']],axis = 1),[-1])
#         self.end_time_windows = tf.reshape(tf.concat([tf.constant([[1.0]]),params['end_windows']],axis = 1),[-1])
#         self.node_cap = params['node_cap']
#         self.vehicle_cap = params['truck_capacities']


#         self.embedder = GraphAttentionEncoder(input_dim=self.embedding_dim,
#                                               num_heads=self.n_heads,
#                                               num_layers=self.n_encode_layers,
#                                               feed_forward_hidden = self.feed_forward_hidden
#                                               )

#         self.decoder = GraphAttentionDecoder(num_heads=self.n_heads,
#                                              output_dim=self.embedding_dim,
#                                              tanh_clipping = self.tanh_clipping,
#                                              decode_type = self.decode_type)

#     def set_decode_type(self, decode_type):
#         self.decode_type = decode_type


#     def _calc_log_likelihood(self, _log_p, a):

#         # Get log_p corresponding to selected actions
#         log_p = tf.gather_nd(_log_p, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2)

#         # Calculate log_likelihood
#         return tf.reduce_sum(log_p,1)

#     def call(self, input, return_pi = False):

#         embeddings, mean_graph_emb = self.embedder(input)
        
#         # ei = tf.reshape(tf.concat([tf.constant([[0.0]]),input[3]],axis = 1),[-1])
#         # li = tf.reshape(tf.concat([tf.constant([[0.0]]),input[4]],axis = 1),[-1])
#         ei = tf.concat([tf.repeat(tf.constant([[0.0]]),tf.shape(input[3])[0],0),input[3]],axis = 1)[0]
#         li = tf.concat([tf.repeat(tf.constant([[0.0]]),tf.shape(input[3])[0],0),input[4]],axis = 1)[0]

#         _log_p, pi,vehicle_used,cost2,distance_cost2,earliness_cost2,tardiness_cost2 = self.decoder(inputs=input, embeddings=embeddings, context_vectors=mean_graph_emb, node_cap=self.node_cap, vehicle_cap=self.vehicle_cap,duration_tensor = self.duration_tensor,distance_tensor = self.distance_tensor)
#         # cost,  distance_cost, earliness_cost, tardiness_cost = self.problem.get_costs(pi, vehicle_used,self.distance_tensor, self.duration_tensor, self.truck_tensor, self.start_time_windows, self.end_time_windows)

#         cost,  distance_cost, earliness_cost, tardiness_cost = self.problem.get_costs(pi, vehicle_used,self.distance_tensor, self.duration_tensor, self.truck_tensor,ei, li)
#         ll = self._calc_log_likelihood(_log_p, pi)

#         # print(earliness_cost,tardiness_cost)
#         # print(earliness_cost2,tardiness_cost2,"herhere")
#         # cost2 = tf.reshape(cost2,[tf.shape(cost2)[0]])
#         # print("OLD COST             ",cost,cost2)
#         if return_pi:
#             return cost, ll, pi,distance_cost, earliness_cost, tardiness_cost

#         return cost, ll,  distance_cost, earliness_cost, tardiness_cost





# def copy_of_tf_model(model, params):
#     """Copy model weights to new model
#     """
#     # https://stackoverflow.com/questions/56841736/how-to-copy-a-network-in-tensorflow-2-0


#     start_window =  tf.repeat(tf.reshape(params['start_windows'],[1,-1]), repeats = 2, axis = 0)
#     end_window = tf.repeat(tf.reshape(params['end_windows'],[1,-1]), repeats = 2, axis = 0)

#     # start_window = tf.repeat(tf.reshape(params['locations_list'][2],[1,-1]),repeats = 2, axis = 0)
#     # end_window = tf.repeat(tf.reshape(params['locations_list'][3],[1,-1]),repeats = 2, axis = 0)

#     # start_window =  tf.repeat(params['locations_list'][2], repeats = 2, axis = 0)
#     # print(start_window)
#     # end_window = tf.repeat(params['locations_list'][3], repeats = 2, axis = 0)

#     data_random = [tf.random.uniform((2, 2,), minval=0, maxval=1, dtype=tf.dtypes.float32),
#                    tf.random.uniform((2, params['graph_size'], 2), minval=0, maxval=1, dtype=tf.dtypes.float32),
#                    tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(2, params['graph_size']),
#                                              dtype=tf.int32), tf.float32) / tf.cast(params['capacities'][params['graph_size']], tf.float32),
#                    start_window, end_window]


#     new_model = AttentionModel(params)
#     set_decode_type(new_model, "sampling")
#     print("RANDOM WEIGHTS ......")
#     _, _ ,_,_,_= new_model(data_random)

#     for a, b in zip(new_model.variables, model.variables):
#         a.assign(b)

#     return new_model

# def rollout(model, dataset, params, disable_tqdm = False):
#     # Evaluate model in greedy mode
#     set_decode_type(model, "greedy")
#     costs_list = []
#     distance_list = []
#     earliness_list = []
#     tardiness_list = []
#     # batch_size = params['batch'] if mode == 'train' else params['val_batch_size']
#     # n_batches = params['n_rollout_batches'] if mode == 'train' else params['n_val_batches']
#     # print("@@@@",mode,"\n")
#     for batch in dataset:
        
#         cost, _, distance_cost,earliness_cost, tardiness_cost  = model(batch)
#         costs_list.append(cost)
#         distance_list.append(distance_cost)
#         earliness_list.append(earliness_cost) 
#         tardiness_list.append(tardiness_cost)

#     return tf.concat(costs_list, axis=0), tf.concat(distance_list, axis=0), tf.concat(earliness_list, axis=0), tf.concat(tardiness_list, axis=0)


# def validate(dataset, model, params):
#     """Validates model on given dataset in greedy mode
#     """
#     val_costs, distance_cost,earliness_cost, tardiness_cost = rollout(model, dataset, params = params)
#     set_decode_type(model, "sampling")
#     mean_cost = tf.reduce_mean(val_costs)
#     distance_cost = tf.reduce_mean(distance_cost)
#     earliness_cost = tf.reduce_mean(earliness_cost)
#     tardiness_cost = tf.reduce_mean(tardiness_cost)
#     return mean_cost, distance_cost, earliness_cost, tardiness_cost


# class RolloutBaseline:

#     def __init__(self, model, params
#                  ):
#         """
#         Args:
#             model: current model
#             filename: suffix for baseline checkpoint filename
#             from_checkpoint: start from checkpoint flag
#             path_to_checkpoint: path to baseline model weights
#             wp_n_epochs: number of warm-up epochs
#             epoch: current epoch number
#             num_samples: number of samples to be generated for baseline dataset
#             warmup_exp_beta: warmup mixing parameter (exp. moving average parameter)
#         """

#         self.num_samples = params['dataset_sizes']['baseline']
#         self.cur_epoch = params['epoch']
#         self.wp_n_epochs = params['number_of_wp_epochs']
#         self.beta = params['warmup_exp_beta']

#         # controls the amount of warmup
#         self.alpha = 0.0

#         self.running_average_cost = None

#         # Checkpoint params
#         self.filename = params['filename']
#         self.from_checkpoint = params['from_checkpoint']
#         self.path_to_checkpoint = params['path_to_checkpoint']
#         self.logs_directory = params['logs_directory']

#         # Problem params
#         self.embedding_dim = params['embedding_dim']
#         self.graph_size = params['graph_size']

#         # create and evaluate initial baseline
#         self._update_baseline(model, self.cur_epoch, params)


#     def _update_baseline(self, model, epoch, params):

#         # Load or copy baseline model based on self.from_checkpoint condition
#         if self.from_checkpoint and self.alpha == 0:
#             print('Baseline model loaded')
#             self.model = load_tf_model(params['path_to_checkpoint'],params)
#         else:
#             self.model = copy_of_tf_model(model,params)


#             # For checkpoint #TODO this was not commented out
#             self.model.save_weights(self.logs_directory + '/baseline_checkpoint_epoch_{}_{}.h5'.format(epoch, self.filename), save_format='h5')

#         # We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
#         # self.dataset = generate_data_onfly(params)
#         self.dataset = baseline_input_fn(params)

#         print(f"Evaluating baseline model on baseline dataset (epoch = {epoch})")
#         self.bl_vals,_,_,_ = rollout(self.model, self.dataset,params = params)
#         self.mean = tf.reduce_mean(self.bl_vals)
#         self.cur_epoch = epoch

#     def ema_eval(self, cost):
#         """This is running average of cost through previous batches (only for warm-up epochs)
#         """

#         if self.running_average_cost is None:
#             self.running_average_cost = tf.reduce_mean(cost)
#         else:
#             self.running_average_cost = self.beta * self.running_average_cost + (1. - self.beta) * tf.reduce_mean(cost)

#         return self.running_average_cost

#     def eval(self, batch, cost):
#         """Evaluates current baseline model on single training batch
#         """

#         if self.alpha == 0:
#             return self.ema_eval(cost)

#         if self.alpha < 1:
#             v_ema = self.ema_eval(cost)
#         else:
#             v_ema = 0.0

#         v_b, _ ,_,_,_= self.model(batch)

#         v_b = tf.stop_gradient(v_b)
#         v_ema = tf.stop_gradient(v_ema)

#         # Combination of baseline cost and exp. moving average cost
#         return self.alpha * v_b + (1 - self.alpha) * v_ema

#     def eval_all(self,dataset, params):
#         """Evaluates current baseline model on the whole dataset only for non warm-up epochs
#         """

#         if self.alpha < 1:
#             return None

#         val_costs,_,_,_ = rollout(self.model, dataset,params=params)

#         return val_costs

#     def epoch_callback(self, model, epoch, params):
#         """Compares current baseline model with the training model and updates baseline if it is improved
#         """ 

#         self.cur_epoch = epoch

#         print(f"Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})")
#         candidate_vals, _,_,_ = rollout(model, self.dataset, params = params)  # costs for training model on baseline dataset
#         candidate_mean = tf.reduce_mean(candidate_vals)

#         diff = candidate_mean - self.mean

#         print(f"Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline epoch {self.cur_epoch} mean {self.mean}, difference {diff}")

#         if diff < 0:
#             # statistic + p-value
#             # print(f"my candidate shape {len(candidate_vals)}")
#             # print(f"my vals shape {len(self.bl_vals)}")
#             t, p = ttest_rel(candidate_vals, self.bl_vals)

#             p_val = p / 2
#             print(f"p-value: {p_val}")

#             if p_val < 0.05:
#                 print('Update baseline')
#                 self._update_baseline(model, self.cur_epoch, params)

#         # alpha controls the amount of warmup
#         if self.alpha < 1.0:
#             self.alpha = tf.cast((self.cur_epoch + 1), tf.float32) / float(self.wp_n_epochs)
#             print(f"alpha was updated to {self.alpha}")


# # def load_tf_model(path, params):
# #     """Load model weights from hd5 file
# #     """
# #     # https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
# #     data_random = [tf.random.uniform((2, 2,), minval=0, maxval=1, dtype=tf.dtypes.float32),
# #                    tf.random.uniform((2, params['graph_size'], 2), minval=0, maxval=1, dtype=tf.dtypes.float32),
# #                    tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(2, params['graph_size']),
# #                                              dtype=tf.int32), tf.float32) / tf.cast(params['capacities'][params['graph_size']], tf.float32)]

# #     model_loaded = AttentionModel(params)
# #     set_decode_type(model_loaded, "greedy")
# #     _, _ = model_loaded(data_random)

# #     model_loaded.load_weights(path)

# #     return model_loaded


# def load_tf_model(path,params):
#     """Load model weights from hd5 file    """

#     model_loaded = AttentionModel(params)


#     start_window =  tf.repeat(tf.reshape(params['start_windows'],[1,-1]), repeats = 2, axis = 0)
#     end_window = tf.repeat(tf.reshape(params['end_windows'],[1,-1]), repeats = 2, axis = 0)

#     print("RANDOM WEIGHTS ......")
#     data_random = [tf.random.uniform((2, 2,), minval=0, maxval=1, dtype=tf.dtypes.float32),
#                    tf.random.uniform((2, params['graph_size'], 2), minval=0, maxval=1, dtype=tf.dtypes.float32),
#                    tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(2, params['graph_size']),
#                                              dtype=tf.int32), tf.float32) / tf.cast(params['capacities'][params['graph_size']], tf.float32),
#                    start_window, end_window]
#     set_decode_type(model_loaded, "greedy")
#     _, _,_,_,_ = model_loaded(data_random)

#     model_loaded.load_weights(path)
# #     print(model.summary())
#     return model_loaded



# def model_skeleton( params):
#     #CHECK POINT 
#     if params['from_checkpoint']:        
#         model_tf = load_tf_model(path = params['model_checkpoint'],params = params)
#     else:
#         # Initialize model
#         model_tf = AttentionModel(params)


#     set_decode_type(model_tf, "sampling")


#     # Initialize baseline
#     baseline = RolloutBaseline(model_tf, params)

#     # Initialize optimizer
#     # https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/

#     optimizer = tf.keras.optimizers.Adam(params['learning_rate'])

#     return optimizer, model_tf, baseline


