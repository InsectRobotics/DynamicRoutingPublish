from collections import deque

class RewardAverage:

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.average_reward = 0
        self.i_episode = 0
        self.reward_sum = 0
        self.episode_reward_deque = deque(maxlen=self.window_size)

    def step(self, episode_reward):
        if self.i_episode >= self.window_size:
            self.reward_sum += episode_reward - self.episode_reward_deque.popleft()
            self.episode_reward_deque.append(episode_reward)
            self.average_reward = self.reward_sum / 100
        else:
            self.reward_sum += episode_reward
            self.average_reward = self.reward_sum / (self.i_episode + 1)
            self.episode_reward_deque.append(episode_reward)
        self.i_episode += 1
        return self.average_reward