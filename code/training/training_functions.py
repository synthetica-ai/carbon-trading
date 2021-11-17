import tensorflow as tf
import numpy as np

from os import makedirs
from os.path import exists

from data.data_functions import train_input_fn
from env.env import CarbonEnv
from models.models import BaselineNet, PolicyNet
from utils.utils import generate_state_at_new_day, prepare_ships_log


class PolicyGradient(object):
    def __init__(self, env, num_iterations=5, output_path="../results/"):
        self.output_path = output_path
        if not exists(output_path):
            makedirs(output_path)
        self.env = env
        self.batch_size = 4  # mallon einai poso megalo trajectory (plh8os steps pou kanw) # self.env.batch_size  # 32
        # to observation shape einai 4, 10+11+1+1 ? 10=contracts_feats,11=fleet_feats,1=contacts_mask_feats,1=fleet_mask_feats
        # self.observation_dim = self.env.observation_space_dim
        # self.action_dim = self.env.action_space_dim[0]
        self.action_dim = 13
        self.gamma = 0.7  # 0.99
        # posa games / years tha trexw
        self.num_iterations = num_iterations
        self.max_ep_len = 365 * 4  # an ka8e mera exw 4 available ships
        self.policy_net = PolicyNet(embedding_size=128, output_size=self.action_dim)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.baseline_net = BaselineNet(embedding_size=128, output_size=1)

    def play_games(self, current_episode, env=None):
        # t = 0
        if env is None:
            env = self.env

        # while num_episodes or t < self.batch_size:
        state = env.reset()  #

        states, actions, rewards = [], [], []

        steps_count = 0
        for day in range(365):

            print(f"Xronia: {current_episode} kai mera: {day}")
            # print(f"To ships_log einai {env.ships_log}")
            # gia ka8e mera ektos ths prwths
            if day != 0:
                env.ships_log, env.available_ships_list = prepare_ships_log(env.ships_log)

            # an h available ships list einai empty
            if not env.available_ships_list:
                print("Den exw available ships ara paw sthn epomenh mera")
                continue
            else:
                print(f"Ta available ships einai {env.available_ships_list}")
                state = generate_state_at_new_day(env, env.available_ships_list)

            # kane concat tous tensores twn contracts_state
            # yearly_contracts_state = states_yearly_dict["contracts_state"]
            # daily_contracts_state = state["contracts_state"]
            # concatenated_contracts_state = tf.concat((yearly_contracts_state, daily_contracts_state), axis=0)
            # states_yearly_dict["yearly_contracts_state"] = concatenated_contracts_state

            # for key in state:
            #     # print(key)
            #     states_yearly_dict[key] = tf.concat((states_yearly_dict[key], state[key]), axis=0)

            for ship_number in env.available_ships_list:

                action = self.policy_net.sample_action(state)
                state, reward, done, _ = env.step(ship_number, action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                steps_count += 1

                # if done or steps_count == self.max_ep_len - 1:
                #     episode_rewards.append(episode_reward)
                #     break
                # if (not num_episodes) and t == self.batch_size:
                #     break

        print(f"Xronia: {current_episode}, sunolo apo steps: {steps_count} ")
        year_data = {"states": states, "reward": rewards, "action": actions}

        return year_data, steps_count

    def get_returns(self, rewards_array):
        T = len(rewards_array)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[: T - t] * rewards_array[t:]) for t in range(T)])
        return returns

    def get_advantage(self, returns, states):
        value_array = np.array([])
        for state in states:
            value = self.baseline_net.forward(state).numpy()
            value_array = np.append(value_array, value)
        advantages = returns - value_array
        advantages = (advantages - np.mean(advantages)) / np.sqrt(np.sum(advantages ** 2))
        return advantages

    def update_policy(self, state, action, advantage, entropy_loss_weight=0.001):
        # state is already a tensor
        action = tf.convert_to_tensor(action)
        advantage = tf.convert_to_tensor(advantage)
        with tf.GradientTape() as tape:
            log_prob = self.policy_net.action_distribution(state)[1].log_prob(action)
            entropy = self.policy_net.action_distribution(state)[1].entropy()
            loss = -tf.math.reduce_mean(
                log_prob * tf.cast(advantage, tf.float32) + entropy_loss_weight * entropy
            )
        grads = tape.gradient(loss, self.policy_net.policy_model.trainable_weights)
        self.policy_optimizer.apply_gradients(
            zip(grads, self.policy_net.policy_model.trainable_weights)
        )
        return loss

    def train(self):
        each_year_reward_list = []
        each_year_avg_reward_list = []
        each_year_actions_list = []
        each_year_baseline_loss_list = []
        each_year_policynet_loss_list = []

        for year in range(self.num_iterations):
            print(f"Ksekina to year: {year}")
            year_data, num_steps_current_year = self.play_games(current_episode=year)
            # o rewards array exei length iso me num_steps_for_current_year
            rewards_array_current_year = year_data["reward"]
            total_reward_current_year = sum(rewards_array_current_year)
            print(f"To synoliko reward gia to year {year} htan {total_reward_current_year}")
            each_year_reward_list.append(total_reward_current_year)

            states_array_current_year = year_data["states"]
            actions_array_current_year = year_data["action"]
            returns_array_current_year = self.get_returns(rewards_array_current_year)
            advantages_array_current_year = self.get_advantage(
                returns_array_current_year, states_array_current_year
            )
            baseline_loss_array_current_year = np.array([])
            policynet_loss_array_current_year = np.array([])
            for step in range(num_steps_current_year):
                state_for_current_step = states_array_current_year[step]
                return_for_current_step = returns_array_current_year[step]
                action_for_current_step = actions_array_current_year[step]
                advantage_for_current_step = advantages_array_current_year[step]
                baseline_loss_array_current_year = np.append(
                    baseline_loss_array_current_year,
                    self.baseline_net.update(
                        state_dict=state_for_current_step, target=return_for_current_step
                    ),
                )
                policynet_loss_array_current_year = np.append(
                    policynet_loss_array_current_year,
                    self.update_policy(
                        state=state_for_current_step,
                        action=action_for_current_step,
                        advantage=advantage_for_current_step,
                    ),
                )
            each_year_actions_list.append(actions_array_current_year)
            each_year_baseline_loss_list.append(baseline_loss_array_current_year)
            each_year_policynet_loss_list.append(policynet_loss_array_current_year)
            avg_reward = np.mean(returns_array_current_year)
            each_year_avg_reward_list.append(avg_reward)
            print(f"To average reward gia ta {self.num_iterations} years htan {avg_reward}")
        print("Training complete")
        np.save(self.output_path + "actions.npy", each_year_actions_list)
        np.save(self.output_path + "baseline_loss.npy", each_year_baseline_loss_list)
        np.save(self.output_path + "policynet_loss.npy", each_year_policynet_loss_list)
        np.save(self.output_path + "rewards.npy", each_year_avg_reward_list)
        self.baseline_net.baseline_model.save("../results/models/baseline/")
        self.policy_net.policy_model.save("../results/models/policynet/")

    def evaluate(self, env, num_episodes=1):
        paths, rewards = self.play_games(env, num_episodes)
        avg_reward = np.mean(rewards)
        print("Average eval reward: {:04.2f}".format(avg_reward))
        return avg_reward


# region
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
# endregion

