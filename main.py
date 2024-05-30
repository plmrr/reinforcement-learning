import matplotlib.pyplot as plt
from game import Game
from ql import QLearningAgent
from gif import create_gif

def train_and_plot(agent, episodes, title):
    agent.train(episodes=episodes)
    plt.plot([_ for _ in range(episodes)], agent.historic_gold)
    plt.ylabel('Gold')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.savefig('output_files/ql/gold_plot.png')
    plt.show()
    plt.plot([_ for _ in range(episodes)], agent.historic_reward)
    plt.ylabel('Reward')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.savefig('output_files/ql/reward_plot.png')
    plt.show()

game = Game()
q_agent = QLearningAgent(game)
train_and_plot(q_agent, episodes=50000, title="Q-Learning Agent")

create_gif(q_agent.frame_path, 'output_files/ql/q_learning.gif')
