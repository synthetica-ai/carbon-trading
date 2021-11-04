import tensorflow as tf
from tensorflow import keras
import numpy as np
from termcolor import colored
from tqdm import tqdm
from datetime import datetime
from os import rmdir, listdir, remove, makedirs
from os.path import exists
from time import gmtime, strftime
from itertools import product
from random import shuffle
from shutil import copyfile

from data.data_functions import train_input_fn
from env.env import CarbonEnv
from models.models import set_decode_type, validate, model_skeleton, BaselineNet, PolicyNet


class PolicyGradient(object):
    def __init__(self, env, num_iterations=2, batch_size=2000, max_ep_len=365*4, output_path="../results/"):
        self.output_path = output_path
        if not exists(output_path):
            makedirs(output_path)
        self.env = env
        # dhlwsh input
        #self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.gamma = 0.99
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)
        self.policy_net = PolicyNet(emb=128, output_size=self.action_dim)
        self.baseline_net = BaselineNet(emb=128, output_size=1)

    def play_games(self, env=None, num_episodes=10):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        if not env:
            env = self.env

        while (num_episodes or t < self.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                # for each ship
                states.append(state)
                action = self.policy_net.sample_action(np.atleast_2d(state))[0]
                state, reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1

                if (done or step == self.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.batch_size:
                    break

            path = {"observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break
        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = []
            reversed_rewards = np.flip(rewards, 0)
            g_t = 0
            for r in reversed_rewards:
                g_t = r + self.gamma*g_t
                returns.insert(0, g_t)
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def get_advantage(self, returns, observations):
        values = self.baseline_net.forward(observations).numpy()
        advantages = returns - values
        advantages = (advantages-np.mean(advantages)) / \
            np.sqrt(np.sum(advantages**2))
        return advantages

    def update_policy(self, observations, actions, advantages):
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions)
        advantages = tf.convert_to_tensor(advantages)
        with tf.GradientTape() as tape:
            log_prob = self.policy_net.action_distribution(
                observations).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob *
                                        tf.cast(advantages, tf.float32))
        grads = tape.gradient(loss, self.policy_net.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_net.model.trainable_weights))

    def train(self):
        all_total_rewards = []
        averaged_total_rewards = []
        for t in range(self.num_iterations):
            paths, total_rewards = self.play_games()
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate(
                [path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            returns = self.get_returns(paths)
            advantages = self.get_advantage(returns, observations)
            self.baseline_net.update(observations=observations, target=returns)
            self.update_policy(observations, actions, advantages)
            avg_reward = np.mean(total_rewards)
            averaged_total_rewards.append(avg_reward)
            print("Average reward for batch {}: {:04.2f}".format(t, avg_reward))
        print("Training complete")
        np.save(self.output_path + "rewards.npy", averaged_total_rewards)
        # export_plot(averaged_total_rewards, "Reward", "CartPole-v0", self.output_path + "rewards.png")

    def eval(self, env, num_episodes=1):
        paths, rewards = self.play_games(env, num_episodes)
        avg_reward = np.mean(rewards)
        print("Average eval reward: {:04.2f}".format(avg_reward))
        return avg_reward

    def make_video(self):
        env = wrappers.Monitor(self.env, self.output_path+"videos", force=True)
        self.eval(env=env, num_episodes=1)


# ##
# def train(validation_dataset,
#             params,
#             config):


#     grad_norm_clipping = params['grad_norm_clipping']
#     batch_verbose = params['batch_verbose']
#     graph_size = params['graph_size']
#     filename = params['filename']
#     logs_directory = params['logs_directory']


#     optimizer, model_tf, baseline = model_skeleton(params)

#     def rein_loss(model, inputs, baseline, num_batch):
#         """Calculate loss for REINFORCE algorithm
#         """

#         # Evaluate model, get costs and log probabilities
#         cost, log_likelihood,  distance_cost, earliness_cost, tardiness_cost = model(inputs)

#         # Evaluate baseline
#         # For first wp_n_epochs we take the combination of baseline and ema for previous batches
#         # after that we take a slice of precomputed baseline values
#         bl_val = bl_vals[num_batch] if bl_vals is not None else baseline.eval(inputs, cost)
#         bl_val = tf.stop_gradient(bl_val)

#         # Calculate loss
#         reinforce_loss = tf.reduce_mean((cost - bl_val) * log_likelihood)

#         return reinforce_loss, tf.reduce_mean(cost),  tf.reduce_mean(distance_cost), tf.reduce_mean(earliness_cost), tf.reduce_mean(tardiness_cost)

#     def grad(model, inputs, baseline, num_batch):
#         """Calculate gradients
#         """
#         with tf.GradientTape() as tape:
#             loss, cost, distance_cost, earliness_cost, tardiness_cost = rein_loss(model, inputs, baseline, num_batch)
#         return loss, cost, tape.gradient(loss, model.trainable_variables),  distance_cost, earliness_cost, tardiness_cost

#     train_loss_results = []
#     train_cost_results = []
#     val_cost_avg = []
#     best_cost = 1e+30


#     # Training loop
#     for epoch in tf.range(config['epochs']):


#         # # Create dataset on current epoch
#         epoch_time_start = datetime.now()
#         train_dataset = train_input_fn(params)
#         end_datagen_time = datetime.now()
#         train_datagen_time = (end_datagen_time - epoch_time_start).seconds/60.


#         epoch_loss_avg = tf.keras.metrics.Mean()
#         epoch_cost_avg = tf.keras.metrics.Mean()

#         # Skip warm-up stage when we continue training from checkpoint
#         if from_checkpoint and baseline.alpha != 1.0:
#             print('Skipping warm-up mode')
#             baseline.alpha = 1.0

#         # If epoch > wp_n_epochs then precompute baseline values for the whole dataset else None
#         baseline_start_time = datetime.now()
#         bl_vals = baseline.eval_all(train_dataset, params)  # (samples, ) or None
#         bl_vals = tf.reshape(bl_vals, (-1, params['batch_sizes']['train'])) if bl_vals is not None else None # (n_batches, batch) or None
#         baseline_end_time = datetime.now()
#         baseline_eval_time = (baseline_end_time - baseline_start_time).seconds/60.

#         print("Current decode type: {}".format(model_tf.decode_type))

#         for num_batch, x_batch in tqdm(enumerate(train_dataset), desc="batch calculation at epoch {}".format(epoch)):
#             # print(f"My batch {x_batch}")
#             # Optimize the model
#             loss_value, cost_val, grads, distance_cost, earliness_cost, tardiness_cost = grad(model_tf, x_batch, baseline, num_batch)

#             # Clip gradients by grad_norm_clipping
#             init_global_norm = tf.linalg.global_norm(grads)
#             grads, _ = tf.clip_by_global_norm(grads, grad_norm_clipping)
#             global_norm = tf.linalg.global_norm(grads)

#             if num_batch%batch_verbose == 0:
#                 print("grad_global_norm = {}, clipped_norm = {}".format(init_global_norm.numpy(), global_norm.numpy()))

#             optimizer.apply_gradients(zip(grads, model_tf.trainable_variables))

#             # Track progress
#             epoch_loss_avg.update_state(loss_value)
#             epoch_cost_avg.update_state(cost_val)

#         # Update baseline if the candidate model is good enough. In this case also create new baseline dataset
#         baseline.epoch_callback(model_tf, epoch, params)
#         set_decode_type(model_tf, "sampling")


#         train_loss_results.append(epoch_loss_avg.result())
#         train_cost_results.append(epoch_cost_avg.result())

#         # Validate current model
#         val_cost, distance_cost, earliness_cost, tardiness_cost = validate(validation_dataset, model_tf, params)
#         val_cost_avg.append(val_cost)

#         # reduce lr on plateau
#         print(f"My best cost :{best_cost}, val_cost: {val_cost}")
#         improvement = val_cost < best_cost# - params['reduce_lr_delta']

#         if params['lr_scheduler'] == 'reduce_on_plateau':
#             print(f"My learning rate is {optimizer._decayed_lr(tf.float32)} and improvement is {improvement}")
#             print(f"My wait is {wait}")
#             if not improvement:
#                 wait +=1
#                 if wait > params['patience']:
#                     new_lr = max(optimizer.learning_rate * params['reduce_lr_factor'], params['min_lr'])
#                     optimizer.learning_rate = new_lr
#                     print(colored(f"My lr will drop to {optimizer.learning_rate}",'red'))
#                     wait = 0
#         elif params['lr_scheduler'] == 'cyclical':
#             lr_index = epoch%len(params['cyclical_lr_values'])
#             new_lr = params['cyclical_lr_values'][lr_index]
#             optimizer.learning_rate = new_lr
#             print(colored(f"My learning rate is {optimizer._decayed_lr(tf.float32)}",'red'))


#         best_cost = min(val_cost,best_cost)

#         # Simos Actual problem
#         set_decode_type(model_tf, "greedy")
#         cost_, ll, pi,distance_cost_, earliness_cost_, tardiness_cost_ = model_tf(params['tour_0_demand'], return_pi=True)
#         set_decode_type(model_tf, "sampling")


#         print(colored(f"Epoch {epoch} -- Loss: {epoch_loss_avg.result()} Cost: {epoch_cost_avg.result()} Validation cost: {val_cost}\n\n",'green'))

#         epoch_time_end= datetime.now()
#         epoch_time = (epoch_time_end - epoch_time_start).seconds/60
#         print(colored(f"Epoch profiling in minutes-- Total epoch :{epoch_time}, Train datagen : {train_datagen_time}, baseline eval : {baseline_eval_time}",'yellow'))

#     return model_tf, best_cost, distance_cost, earliness_cost, tardiness_cost,cost_, distance_cost_, earliness_cost_, tardiness_cost_
