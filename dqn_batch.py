# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import os

EPISODES = 1000 #number of games for the agent to play

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    #discount rate
        self.epsilon = 1.0  #exploration rate, starting value of epsilon, thanks to epsilon  the agen ranomly decides its action
        self.epsilon_min = 0.01 #minimum value of epsilon, so the agent can explore at least this amount
        self.epsilon_decay = 0.995 #multiplicative factor (per episode) for decreasing epsilon, so the the number of exploration decrease as the agent feels good while playing
        self.learning_rate = 0.001 #step size, how much the neural net learns in each  iteration
        self.model = self._build_model()

    def _build_model(self):
        #Neural Net for Deep-Q learning Model

        #Sequential() creates the layers.
        model = keras.models.Sequential()
        #Input lauer of state size=4 and a Hidden layer with  24 nodes
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        #Hidden layer with 24 nodes
        model.add(keras.layers.Dense(24, activation='relu'))
        #Output layer with # of actions: 2 nodes, left and right
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        #Creates the model based on the previous info
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    #list of previous experiences and observations using memory
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #how the agent acts
    def act(self, state):
        #at the beginning the agent select its action randomly (random between 0 and 1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
    
        act_values = self.model.predict(state) # predict the reward value based on the current state
        return np.argmax(act_values[0])  # returns action based on predicted reward

    #Training with experiences in the memory
    def replay(self, batch_size):
        #Samples of some experiences and call themm in minibatch
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        #Information extracted from each memory
        for state, action, reward, next_state, done in minibatch:
            #if donde = true, target = reward
            target = reward
            if not done:
                #predict the future discounted reward, reward + whats comming in the next state
                #Q-value returns 2 outputs: one left and one right (maximum is Q-value)
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            #with self.model tries to train itself, make the agent to approximately map the current state to future discounted reward
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        #the Neural Net is trained with the state and target_f    
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1') #gym environment initialized
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    #Iterative process
    for e in range(EPISODES):
        state = env.reset() #reset state at the begining of each game
        state = np.reshape(state, [1, state_size])

        for time in range(500): #time represents each frame of the game
            #500 is total score, the goal is keep the pole upright as until we reach 500
            # env.render()

            #decide action
            action = agent.act(state)
            #game advances to the nex frame based on the action, reward is 1  for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            #memorize previous information
            agent.memorize(state, action, reward, next_state, done)
            #next_state becomesthe current state  for next frame
            state = next_state

            #done = true when game ends
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}".format(e, EPISODES, time, loss))  
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
