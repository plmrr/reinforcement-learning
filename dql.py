import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from collections import deque
from tensorflow.keras.losses import Huber # type: ignore
from gif import visualize

tf.random.set_seed(5)

class DQLAgent:
    def __init__(self, game, learning_rate=0.01, discount_rate=0.99, exploration_rate=1.0,
                 exploration_decay_rate=0.0002, exploration_min=0.01, batch_size=128, memory_size=2000000,
                 target_update_frequency=20, frame_path='output_files/dql/frames'):
        self.game = game
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.actions = self.generate_actions()
        self.frame_path = frame_path
        self.historic_gold = []
        self.historic_reward = []
        self.turn = 0
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0

        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def generate_actions(self):
        actions = []
        for unit_id in range(2):
            for action_type in [0, 1]:
                if action_type == 0:
                    for direction in [0, 1, 2, 3]:
                        for steps in range(1, 3):
                            actions.append((unit_id, action_type, direction, steps))
                elif action_type == 1:
                    for city_type in [0, 1]:
                        actions.append((unit_id, action_type, city_type, 0))
        return actions

    def build_model(self):
        model = Sequential()
        model.add(Dense(18, input_dim=7, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.actions), activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=Huber())
        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def encode_state(self, state):
        state_vector = []
        for unit_id in range(2):
            if unit_id < len(state['units']):
                _, position = state['units'][unit_id]
                state_vector += [position[0], position[1]]
            else:
                state_vector += [self.game.board_size, self.game.board_size]
        state_vector += [state['gold'], state['wood'], state['iron']]
        return np.array(state_vector)

    def get_action_from_policy(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(self.actions)
        state_vector = self.encode_state(state).reshape(1, -1)
        q_values = self.model.predict(state_vector, verbose=0)
        return self.actions[np.argmax(q_values[0])]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([self.encode_state(state) for state, _, _, _, _ in minibatch])
        next_states = np.array([self.encode_state(next_state) for _, _, _, next_state, _ in minibatch])
        q_values = self.model.predict(states, verbose=0)
        q_next_values = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.discount_rate * np.amax(q_next_values[i])
            action_index = self.actions.index(action)
            q_values[i][action_index] = target

        self.model.fit(states, q_values, epochs=1, verbose=0, batch_size=self.batch_size)

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()

        tf.keras.backend.clear_session()  # Wyczyść sesję

    def train(self, episodes):
        for episode in range(episodes):
            state = self.game.reset()
            done = False
            self.turn = 0
            total_reward = 0
            while not done:
                if episode == episodes - 1:  # visualize only in the last episode
                    visualize(state, self.turn, self.frame_path, self.game.board_size)
                action = self.get_action_from_policy(state)
                next_state, reward, done = self.game.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.turn += 1
            self.replay()
            self.historic_reward.append(total_reward)
            self.historic_gold.append(state['gold'])
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate = self.exploration_rate * (1 - self.exploration_decay_rate)
            print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}, Exploration Rate: {self.exploration_rate}")

            tf.keras.backend.clear_session()  # Wyczyść sesję
