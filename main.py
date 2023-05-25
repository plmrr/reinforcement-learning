import numpy as np
import random
import matplotlib.pyplot as plt
from game import Game


class QLearningAgent:
    def __init__(self, game, learning_rate=0.1, discount_rate=0.99, exploration_rate=1.0, exploration_decay_rate=0.001):
        self.game = game
        self.epochs = 10000
        self.historic_gold = []
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = self.generate_actions()
        self.q_table = np.zeros((self.game.board_size**4 * 3**3 * 10**3, len(self.actions)))

    def generate_actions(self):
        actions = []
        for unit_id in range(2):
            for action_type in ['move', 'build']:
                if action_type == 'move':
                    for direction in ['up', 'down', 'left', 'right']:
                        for steps in range(1, 3):
                            actions.append((unit_id, action_type, direction, steps))
                elif action_type == 'build':
                    for city_type in ['basic', 'wood', 'iron']:
                        actions.append((unit_id, action_type, city_type, 0))
        return actions

    def encode_state(self, state):
        state_vector = []
        for unit_id in range(2):
            if unit_id < len(state['units']):
                unit_type, position = state['units'][unit_id]
                state_vector += [unit_type, position[0], position[1]]
            else:
                state_vector += [0, self.game.board_size, self.game.board_size]
        state_vector += [state['gold'], state['wood'], state['iron']]
        state_index = 0
        for i, value in enumerate(state_vector):
            state_index += value * (self.game.board_size + 1)**i
        return state_index

    def get_action_from_policy(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:
            state_index = self.encode_state(state)
            return self.actions[np.argmax(self.q_table[state_index])]

    def update_q_table(self, old_state, action, reward, new_state):
        old_state_index = self.encode_state(old_state)
        new_state_index = self.encode_state(new_state)
        action_index = self.actions.index(action)
        self.q_table[old_state_index, action_index] = (1 - self.learning_rate) * self.q_table[old_state_index, action_index] + \
            self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state_index, :]))

    def train(self, num_episodes=10000):
        for _ in range(num_episodes):
            state = self.game.reset()
            print('state', state)
            done = False
            while not done:
                self.game.print_board()
                action = self.get_action_from_policy(state)
                print('action', action)
                new_state, reward, done = self.game.step(action)
                print('new_state', new_state)
                print('reward', reward)
                if 'error' not in new_state:
                    self.update_q_table(state, action, reward, new_state)
                    state = new_state
                else:
                    self.update_q_table(state, action, -10, state)
            self.historic_gold.append(state['gold'])
            self.exploration_rate = self.exploration_rate * (1 - self.exploration_decay_rate)


game = Game()
agent = QLearningAgent(game)
agent.train()
plt.plot([_ for _ in (range(agent.epochs))], agent.historic_gold)
plt.ylabel('Gold')
plt.xlabel('Epoch')
plt.show()
