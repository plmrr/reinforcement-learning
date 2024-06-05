import matplotlib.pyplot as plt
from game import Game
from ql import QLearningAgent
from dql import DQLAgent
from gif import create_gif
import tensorflow as tf
import os


def train_and_plot(agent, episodes, title, path):
    agent.train(episodes=episodes)
    plt.plot([_ for _ in range(episodes)], agent.historic_gold)
    plt.ylabel('Gold')
    plt.xlabel('Epoch')
    plt.title(title)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/gold_plot.png')
    plt.show()
    plt.plot([_ for _ in range(episodes)], agent.historic_reward)
    plt.ylabel('Reward')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.savefig(f'{path}/reward_plot.png')
    plt.show()


game = Game()
ql_path = 'output_files/ql'
q_agent = QLearningAgent(game, frame_path=f'{ql_path}/frames')
train_and_plot(q_agent, episodes=5000, title="Q-Learning Agent", path=ql_path)

# create_gif(q_agent.frame_path, f'{ql_path}/q_learning.gif')

# game = Game()
# dql_path = 'output_files/dql'
# dqn_agent = DQLAgent(game, frame_path=f'{dql_path}/frames')
# train_and_plot(dqn_agent, episodes=10000, title="DQN Agent", path=dql_path)

# create_gif(dqn_agent.frame_path, f'{dql_path}/dqn.gif')
