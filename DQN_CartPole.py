# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 00:24:58 2019

@author: Unnikrishnan Menon
"""

import gym
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense

env = gym.make('CartPole-v1')

model=Sequential()
model.add(Dense(24,input_dim=env.observation_space.shape[0],activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(env.action_space.n,activation='relu'))
model.compile(optimizer='Adam',loss='mse',metrics=['mae'])

gamma = 1.0
epsilon = 1.0
m = []

for i in range(5000):
    state = env.reset()
    state = np.array([state])
    for t in range(500):
        if np.random.rand() <= epsilon:
            action = random.randrange(env.action_space.n)
        else:
            action = np.argmax(model.predict(state))
        next_state,reward,done,observation = env.step(action)
        next_state = np.array([next_state])
        tot = reward + gamma * np.max(model.predict(next_state))
        p = model.predict(state)[0]
        p[action] = tot
        model.fit(state, p.reshape(-1, env.action_space.n), epochs=1, verbose=0)
        m.append((state,action,reward,next_state,done))
        state = next_state
        if done:
            print("Episode : {}, Score: {}".format(i,t))
            break
        if len(m)==50000:
            del m[:5000]
    if epsilon > 0.01:
        epsilon *= 0.999
    if len(m) > 64:
        for state, action, reward, next_state, done in random.sample(m,64):
            tot=reward
            if not done:
              tot=reward + gamma * np.max(model.predict(next_state))

            p = model.predict(state)[0]
            p[action] = tot
            model.fit(state,p.reshape(-1,env.action_space.n), epochs=1, verbose=0)

for i in range(20):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = np.argmax(model.predict(np.array([state])))
        next_state, reward, done, observation = env.step(action)
        state = next_state