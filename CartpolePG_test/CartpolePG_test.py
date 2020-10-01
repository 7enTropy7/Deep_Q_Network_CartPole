# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:45:37 2020

@author: ACER
""" 
    
import gym
import numpy as np
import os
from gym import wrappers
from keras import layers
from keras.utils import to_categorical
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
import matplotlib.pyplot as plt
from gym import wrappers

from MountainCar_RBF import plot_running_avg

class PolicyModel(object):

    def __init__(self, input_dim, output_dim, hidden_dims):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.i = layers.Input(shape=(input_dim,))
        net = self.i

        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("tanh")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)

        self.model = Model(inputs=self.i, outputs=net)
        
        pa_s = self.model.output
        one_hot_action_taken = K.placeholder(shape=(None, self.output_dim),name="action_onehot")
        
        returns = K.placeholder(shape=(None,),name="discount_reward")
        
        action_prob = K.sum(pa_s * one_hot_action_taken, axis=1)
        loss = - (K.log(action_prob) * returns)
        loss = K.mean(loss)
        opt = optimizers.Adagrad()
        updates = opt.get_updates(params=self.model.trainable_weights,loss=loss)
        
        self.fit_fn =  K.function(inputs=[self.model.input,
                                           one_hot_action_taken,
                                           returns],
                                   outputs=[self.model.output],
                                   updates=updates)
        
    def sample_action(self, state):
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.input_dim,), "{} != {}".format(shape, self.input_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.input_dim), "{} != {}".format(shape, self.input_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.model.predict(state))
        #print(action_prob)
        #print(self.output_dim)
        assert len(action_prob) == self.output_dim, "{} != {}".format(len(action_prob), self.output_dim)
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

    def partial_fit(self, S, A, R):
        #print(A)
        onehot_action_taken = np_utils.to_categorical(A, num_classes=self.output_dim)
        returns = R#compute_discounted_R(R)

        assert S.shape[1] == self.input_dim, "{} != {}".format(S.shape[1], self.input_dim)
        assert onehot_action_taken.shape[0] == S.shape[0], "{} != {}".format(onehot_action_taken.shape[0], S.shape[0])
        assert onehot_action_taken.shape[1] == self.output_dim, "{} != {}".format(onehot_action_taken.shape[1], self.output_dim)
        assert len(returns.shape) == 1, "{} != 1".format(len(returns.shape))

        self.fit_fn([S, onehot_action_taken, returns])
        

def compute_discounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    #print(type(R),len(R))
    running_add = 0
        
    for t in range(len(R)):
        running_add = R[t]
        discounted_r[t] = running_add

    #discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r    


def run_episode(env, model):
    """Returns an episode reward
    (1) Play until the game is done
    (2) The agent will choose an action according to the policy
    (3) When it's done, it will train from the game play
    Args:
        env (gym.env): Gym environment
        agent (Agent): Game Playing Agent
    Returns:
        total_reward (int): total reward earned during the whole episode
    """
    done = False
    States = []
    Actions = []
    Rewards = []
    
    gamma = 0.99

    s = env.reset()

    total_reward = 0

    while not done:

        a = model.sample_action(s)

        s2, r, done, info = env.step(a)
        total_reward += r

        States.append(s)
        Actions.append(a)
        Rewards.append(r)

        s = s2

        if done:
            States = np.array(States)
            Actions = np.array(Actions)
            Rewards = np.array(Rewards)
            
            returns = np.zeros_like(Rewards, dtype=np.float32)
            G = 0
            for t in reversed(range(len(Rewards))):
        
                G = G * gamma + Rewards[t]
                returns[t] = G

            returns -= returns.mean() / returns.std()
            

            model.partial_fit(States, Actions, returns)

    return total_reward


def main():
    try:
        env = gym.make("CartPole-v0")
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = PolicyModel(input_dim, output_dim, [16, 16])
        total_rewards = [] 
        
        

        for episode in range(2000):
            reward = run_episode(env, agent)
            total_rewards.append(reward)
            print(episode, reward)
            if episode == 1997:
                filename = os.path.basename(__file__).split('.')[0]
                monitor_dir = './' + filename #+ '_' + str(datetime.now())
                env = wrappers.Monitor(env, monitor_dir,video_callable=lambda episode_id: True,force=True)

        plt.plot(total_rewards)
        plt.show()

    finally:
        env.close()


if __name__ == '__main__':
    main()


  


    
    
    
        
    
        
        