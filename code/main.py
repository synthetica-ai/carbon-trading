import gym

env = gym.make("CartPole-v1")
obs = env.reset()
print(obs)
