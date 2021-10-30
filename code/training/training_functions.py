import tensorflow as tf
from tensorflow import keras
import numpy as np
from termcolor import colored
from tqdm import tqdm
from datetime import datetime
from os import rmdir, listdir, remove
from os.path import exists
from time import gmtime, strftime
from itertools import product
from random import shuffle
from shutil import copyfile

from data.data_functions import train_input_fn
from env.env import CarbonEnv
from models.models import set_decode_type,validate, model_skeleton



# # compute grads of a single action-step
# def play_one_step(env, obs, model, loss_fn):
#     with tf.GradientTape() as tape:
#         left_proba = model(obs[np.newaxis])  # model proba of going left
#         # exploration - exploitation
#         action = (
#             tf.random.uniform([1, 1]) > left_proba
#         )  # action left (0) with prob left_proba or right (1) with prob 1-left_proba
#         y_target = tf.constant([[1.0]]) - tf.cast(
#             action, tf.float32
#         )  # target prob of going left
#         loss = tf.reduce_mean(loss_fn(y_target, left_proba))
#     grads = tape.gradient(loss, model.trainable_variables)
#     obs, reward, done, info = env.step(int(action[0, 0].numpy()))
#     return obs, reward, done, grads


# # play multiple episodes/games, returning all the rewards and gradients for each episode
# def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
#     all_rewards = []
#     all_grads = []
#     for episode in range(n_episodes):
#         current_rewards = []
#         current_grads = []
#         obs = env.reset()
#         for step in range(n_max_steps):
#             obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
#             current_rewards.append(reward)
#             current_grads.append(grads)
#             if done:
#                 break
#         all_rewards.append(current_rewards)
#         all_grads.append(current_grads)
#     return all_rewards, all_grads


# def discount_rewards(rewards, gamma):
#     discounted = np.array(rewards)
#     for step in range(len(rewards) - 2, -1, -1):
#         discounted[step] += discounted[step + 1] * gamma
#     return discounted


# def discount_and_normalize_rewards(all_rewards, gamma):
#     all_discounted_rewards = [
#         discount_rewards(rewards, gamma) for rewards in all_rewards
#     ]
#     flat_rewards = np.concatenate(all_discounted_rewards)
#     reward_mean = flat_rewards.mean()
#     reward_std = flat_rewards.std()
#     return [
#         (discounted_rewards - reward_mean) / reward_std
#         for discounted_rewards in all_discounted_rewards
#     ]


# # hyperparams

# n_iterations = 150
# n_episodes_per_update = 10
# n_max_steps = 200
# gamma = 0.95

# optimizer = keras.optimizers.Adam(lr=0.01)
# loss_fn = keras.losses.binary_crossentropy


# for iteration in range(n_iterations):
#     all_rewards, all_grads = play_multiple_episodes(
#         env, n_episodes_per_update, n_max_steps, model, loss_fn
#     )

#     # how good or bad an action was
#     all_final_rewards = discount_and_normalize_rewards(all_rewards, gamma)
#     all_mean_grads = []
#     for var_index in range(len(model.trainable_variables)):
#         mean_grads = tf.reduce_mean(
#             [
#                 final_reward * all_grads[episode_index][step][var_index]
#                 for episode_index, final_rewards in enumerate(all_final_rewards)
#                 for step, final_reward in enumerate(final_rewards)
#             ],
#             axis=0,
#         )
#         all_mean_grads.append(mean_grads)
#     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))




## 
def train(validation_dataset,
            params, 
            config):


    
    grad_norm_clipping = params['grad_norm_clipping']
    batch_verbose = params['batch_verbose']
    graph_size = params['graph_size']
    filename = params['filename']
    logs_directory = params['logs_directory']


    optimizer, model_tf, baseline = model_skeleton(params)

    def rein_loss(model, inputs, baseline, num_batch):
        """Calculate loss for REINFORCE algorithm
        """

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood,  distance_cost, earliness_cost, tardiness_cost = model(inputs)

        # Evaluate baseline
        # For first wp_n_epochs we take the combination of baseline and ema for previous batches
        # after that we take a slice of precomputed baseline values
        bl_val = bl_vals[num_batch] if bl_vals is not None else baseline.eval(inputs, cost)
        bl_val = tf.stop_gradient(bl_val)

        # Calculate loss
        reinforce_loss = tf.reduce_mean((cost - bl_val) * log_likelihood)

        return reinforce_loss, tf.reduce_mean(cost),  tf.reduce_mean(distance_cost), tf.reduce_mean(earliness_cost), tf.reduce_mean(tardiness_cost)

    def grad(model, inputs, baseline, num_batch):
        """Calculate gradients
        """
        with tf.GradientTape() as tape:
            loss, cost, distance_cost, earliness_cost, tardiness_cost = rein_loss(model, inputs, baseline, num_batch)
        return loss, cost, tape.gradient(loss, model.trainable_variables),  distance_cost, earliness_cost, tardiness_cost 

    train_loss_results = []
    train_cost_results = []
    val_cost_avg = []
    best_cost = 1e+30 


    # Training loop
    for epoch in tf.range(config['epochs']):


        # # Create dataset on current epoch
        epoch_time_start = datetime.now()
        train_dataset = train_input_fn(params)
        end_datagen_time = datetime.now()
        train_datagen_time = (end_datagen_time - epoch_time_start).seconds/60.


        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_cost_avg = tf.keras.metrics.Mean()

        # Skip warm-up stage when we continue training from checkpoint
        if from_checkpoint and baseline.alpha != 1.0:
            print('Skipping warm-up mode')
            baseline.alpha = 1.0

        # If epoch > wp_n_epochs then precompute baseline values for the whole dataset else None
        baseline_start_time = datetime.now()
        bl_vals = baseline.eval_all(train_dataset, params)  # (samples, ) or None
        bl_vals = tf.reshape(bl_vals, (-1, params['batch_sizes']['train'])) if bl_vals is not None else None # (n_batches, batch) or None
        baseline_end_time = datetime.now()
        baseline_eval_time = (baseline_end_time - baseline_start_time).seconds/60.

        print("Current decode type: {}".format(model_tf.decode_type))

        for num_batch, x_batch in tqdm(enumerate(train_dataset), desc="batch calculation at epoch {}".format(epoch)):
            # print(f"My batch {x_batch}")
            # Optimize the model
            loss_value, cost_val, grads, distance_cost, earliness_cost, tardiness_cost = grad(model_tf, x_batch, baseline, num_batch)

            # Clip gradients by grad_norm_clipping
            init_global_norm = tf.linalg.global_norm(grads)
            grads, _ = tf.clip_by_global_norm(grads, grad_norm_clipping)
            global_norm = tf.linalg.global_norm(grads)

            if num_batch%batch_verbose == 0:
                print("grad_global_norm = {}, clipped_norm = {}".format(init_global_norm.numpy(), global_norm.numpy()))

            optimizer.apply_gradients(zip(grads, model_tf.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_cost_avg.update_state(cost_val)

        # Update baseline if the candidate model is good enough. In this case also create new baseline dataset
        baseline.epoch_callback(model_tf, epoch, params)
        set_decode_type(model_tf, "sampling")


        train_loss_results.append(epoch_loss_avg.result())
        train_cost_results.append(epoch_cost_avg.result())
        
        # Validate current model
        val_cost, distance_cost, earliness_cost, tardiness_cost = validate(validation_dataset, model_tf, params)
        val_cost_avg.append(val_cost)

        # reduce lr on plateau
        print(f"My best cost :{best_cost}, val_cost: {val_cost}")
        improvement = val_cost < best_cost# - params['reduce_lr_delta'] 

        if params['lr_scheduler'] == 'reduce_on_plateau':
            print(f"My learning rate is {optimizer._decayed_lr(tf.float32)} and improvement is {improvement}")
            print(f"My wait is {wait}")
            if not improvement:
                wait +=1                 
                if wait > params['patience']:
                    new_lr = max(optimizer.learning_rate * params['reduce_lr_factor'], params['min_lr'])
                    optimizer.learning_rate = new_lr
                    print(colored(f"My lr will drop to {optimizer.learning_rate}",'red'))
                    wait = 0
        elif params['lr_scheduler'] == 'cyclical':
            lr_index = epoch%len(params['cyclical_lr_values'])
            new_lr = params['cyclical_lr_values'][lr_index]
            optimizer.learning_rate = new_lr
            print(colored(f"My learning rate is {optimizer._decayed_lr(tf.float32)}",'red'))


        best_cost = min(val_cost,best_cost) 

        # Simos Actual problem
        set_decode_type(model_tf, "greedy")
        cost_, ll, pi,distance_cost_, earliness_cost_, tardiness_cost_ = model_tf(params['tour_0_demand'], return_pi=True)
        set_decode_type(model_tf, "sampling")


        print(colored(f"Epoch {epoch} -- Loss: {epoch_loss_avg.result()} Cost: {epoch_cost_avg.result()} Validation cost: {val_cost}\n\n",'green')) 
        
        epoch_time_end= datetime.now()
        epoch_time = (epoch_time_end - epoch_time_start).seconds/60
        print(colored(f"Epoch profiling in minutes-- Total epoch :{epoch_time}, Train datagen : {train_datagen_time}, baseline eval : {baseline_eval_time}",'yellow'))

    return model_tf, best_cost, distance_cost, earliness_cost, tardiness_cost,cost_, distance_cost_, earliness_cost_, tardiness_cost_


