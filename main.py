import random
from game import Game
import matplotlib.pyplot as plt
import numpy as np
from gif import create_gif


class QLearningAgent:
    def __init__(self, game, learning_rate=0.075, discount_rate=0.99, exploration_rate=1.0, exploration_decay_rate=0.0005):
        self.game = game
        self.epochs = 100000
        self.historic_gold = []
        self.historic_reward = []
        self.reward = 0
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = self.generate_actions()
        self.frame_path = 'frames'
        self.turn = 0
        self.q_table = np.zeros((self.game.board_size**4 * 3**3 * 10**3, len(self.actions)))

    def visualize(self, state):
        color_map = {
            1: [172, 44, 7],  # Settler
            2: [3, 47, 25],  # Scout
            3: [255, 199, 88],  # Basic city
            4: [102, 51, 0],  # Wooden city
            5: [161, 157, 148],  # Iron city
            6: [133, 94, 66],  # Wood
            7: [78, 79, 85],  # Iron
            8: [215, 183, 64]  # Gold
        }

        labels = {
            1: "Settler",
            2: "Scout",
            3: "City",
            4: "W_City",
            5: "I_City",
            6: "Wood",
            7: "Iron",
            8: "Gold"
        }

        board = np.zeros((self.game.board_size, self.game.board_size, 3))
        text = np.full((self.game.board_size, self.game.board_size), "", dtype='U20')
        for unit in state['units']:
            if unit[0]:  # Settler
                board[unit[1][0], unit[1][1]] = color_map[1]
                if text[unit[1][0], unit[1][1]] != "":
                    text[unit[1][0], unit[1][1]] += "+" + labels[1]
                else:
                    text[unit[1][0], unit[1][1]] = labels[1]
            else:  # Scout
                board[unit[1][0], unit[1][1]] = color_map[2]
                if text[unit[1][0], unit[1][1]] != "":
                    text[unit[1][0], unit[1][1]] += "+" + labels[2]
                else:
                    text[unit[1][0], unit[1][1]] = labels[2]

        for city in state['basic_cities']:
            board[city[0], city[1]] = color_map[3]
            if text[city[0], city[1]] != "":
                text[city[0], city[1]] += "+" + labels[3]
            else:
                text[city[0], city[1]] = labels[3]

        for city in state['wooden_cities']:
            board[city[0], city[1]] = color_map[4]
            if text[city[0], city[1]] != "":
                text[city[0], city[1]] += "+" + labels[4]
            else:
                text[city[0], city[1]] = labels[4]

        for city in state['iron_cities']:
            board[city[0], city[1]] = color_map[5]
            if text[city[0], city[1]] != "":
                text[city[0], city[1]] += "+" + labels[5]
            else:
                text[city[0], city[1]] = labels[5]

        for resource in state['resources'].keys():
            resource_type = state['resources'][resource]
            if resource_type == 'wood':
                board[resource[0], resource[1]] = color_map[6]
                text[resource[0], resource[1]] = labels[6]
            elif resource_type == 'iron':
                board[resource[0], resource[1]] = color_map[7]
                text[resource[0], resource[1]] = labels[7]
            elif resource_type == 'gold':
                board[resource[0], resource[1]] = color_map[8]
                text[resource[0], resource[1]] = labels[8]

        board[board.sum(axis=2) == 0] = [96, 221, 73]

        board = board / 255.0  # normalize rgb

        fig, ax = plt.subplots()
        ax.imshow(board)

        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                ax.text(j, i, text[i, j], ha='center', va='center', color='black', fontsize=8)

        ax.set_xticks(np.arange(-.5, self.game.board_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.game.board_size, 1), minor=True)

        ax.grid(which='minor', color='black', linewidth=2)
        plt.title(f'Turn: {self.turn}, gold: {state["gold"]}')
        plt.savefig(f'{self.frame_path}/frame_{str(self.turn).zfill(5)}.png')
        plt.close()

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

    def train(self):
        for episode in range(self.epochs):
            print(episode)
            state = self.game.reset()
            done = False
            self.turn = 0
            self.reward = 0
            while not done:
                if episode == self.epochs - 1:  # visualize episode
                    self.visualize(state)
                action = self.get_action_from_policy(state)
                new_state, reward, done = self.game.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                self.turn += 1
                self.reward += reward
            self.historic_reward.append(self.reward)
            self.historic_gold.append(state['gold'])
            self.exploration_rate = self.exploration_rate * (1 - self.exploration_decay_rate)


game = Game()
agent = QLearningAgent(game)
agent.train()
plt.plot([_ for _ in (range(agent.epochs))], agent.historic_gold)
plt.ylabel('Gold')
plt.xlabel('Epoch')
create_gif(agent.frame_path, 'new.gif')
plt.show()
plt.plot([_ for _ in (range(agent.epochs))], agent.historic_reward)
plt.ylabel('Reward')
plt.xlabel('Epoch')
plt.show()
