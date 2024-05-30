import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from game import Game
from gif import visualize

class DQLAgent:
    def __init__(self, game, learning_rate=0.001, discount_rate=0.99, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01, batch_size=64, memory_size=100000, frame_path='output_files/dql/frames'):
        self.game = game
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []
        self.actions = self.generate_actions()
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.turn = 0
        self.frame_path = frame_path
        self.historic_gold = []
        self.historic_reward = []
        self.reward = 0

        # Ensure the frame directory exists
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)

    def build_model(self):
        input_dim = 2 * 3 + 3  # 2 units * (type + x + y) + 3 resources
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
                state_vector += [1 if isinstance(unit_type, bool) else 0, position[0], position[1]]
            else:
                state_vector += [0, self.game.board_size, self.game.board_size]
        state_vector += [state['gold'], state['wood'], state['iron']]
        return np.array(state_vector).reshape(1, -1)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(self.actions)
        state_vector = self.encode_state(state)
        q_values = self.model.predict(state_vector)
        return self.actions[np.argmax(q_values[0])]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_vector = self.encode_state(next_state)
                target = reward + self.discount_rate * np.amax(self.target_model.predict(next_state_vector)[0])
            state_vector = self.encode_state(state)
            target_f = self.model.predict(state_vector)
            target_f[0][self.actions.index(action)] = target
            self.model.fit(state_vector, target_f, epochs=1, verbose=0)
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

    def train(self, episodes=1000, target_update_freq=10):
        for episode in range(episodes):
            print(episode)
            state = self.game.reset()
            done = False
            self.turn = 0
            self.reward = 0
            while not done:
                if episode == episodes - 1:  # visualize only in the last episode
                    visualize(state, self.turn, self.frame_path, self.game.board_size)
                action = self.act(state)
                next_state, reward, done = self.game.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.turn += 1
                self.reward += reward
                self.replay()
            if episode % target_update_freq == 0:
                self.update_target_model()
            self.historic_reward.append(self.reward)
            self.historic_gold.append(state['gold'])
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
