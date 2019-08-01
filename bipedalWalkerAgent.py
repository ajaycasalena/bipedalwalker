import gym
import numpy as np
from gym.utils import seeding



class Model:
    """
    Class representing the policy
    """

    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size, output_size))

    def predict(self, inp, deltas=None):
        w = self.weights
        if deltas:
            w += deltas
        output = np.dot(inp, w)
        return output

    # returns model weights
    def get_weights(self):
        return self.weights

    # sets model weights
    def set_weights(self, weights):
        self.weights = weights


class Normalizer:
    """
    Normalzies the input to be between 0 and 1
    """

    def __init__(self, input_size):
        self.n = np.zeros(input_size)
        self.mean = np.zeros(input_size)
        self.mean_diff = np.zeros(input_size)
        self.std = np.zeros(input_size)

    # given new data it updates parametest of normalizer
    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    # normalizes the input
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Agent:
    """
    Class used to train model with ARS
    """

    save = True
    load = False

    GAME = 'BipedalWalkerHardcore-v2'

    def __init__(self):
        self.env = gym.make(self.GAME)
        self.env.seed(0)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.shape[0]
        self.model = Model(self.input_size, self.output_size)
        self.normalizer = Normalizer(self.input_size)
        self.noise_rate = 0.06
        self.alpha = 0.09
        self.population = 16
        np.random.seed(1)


    # plays an episode of a game
    def play(self, deltas=None, render=False):
        total_reward = 0
        observation = self.env.reset()
        n = 0
        while n < 2000:
            if render:
                self.env.render()
            self.normalizer.observe(observation)
            observation = self.normalizer.normalize(observation)
            action = self.model.predict(observation, deltas)
            observation, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            total_reward += reward
            n += 1
            if done:
                break
        return total_reward

    def save(self):
        np.save('./bipedalhardcore.npy', self.model.get_weights(), allow_pickle=True)
        print(self.model.get_weights())
        print('progress saved')


    def train(self, n_steps):
        for step in range(n_steps):
            deltas = [self.noise_rate * np.random.randn(*self.model.weights.shape) for _ in range(self.population)]
            positive_rewards = [0] * self.population
            negative_rewards = [0] * self.population
            if self.load:
                old_weights = np.load('bipedal.npy')
                print(old_weights)
                self.load = False
            else:
                old_weights = self.model.get_weights()

            for k, d in enumerate(deltas):
                self.model.set_weights(old_weights + d)
                positive_rewards[k] = self.play(render=False)
                self.model.set_weights(old_weights - d)
                negative_rewards[k] = self.play(render=False)

            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.population]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            update = np.zeros(self.model.weights.shape)
            for r_p, r_n, delta in rollouts:
                update += (r_p - r_n) * delta
            new_weights = old_weights + self.alpha * update / (self.population)
            self.model.set_weights(new_weights)

            re = True
            if step % 20 == 0:
                re = True
            reward = self.play(render=re)
            print('Step: ', step, 'Reward: ', reward)
            if step %(n_steps -1) == 0 and step != 0:
                if self.save:
                    self.save()





if __name__ == '__main__':
    agent = Agent()
    agent.train(1001)
