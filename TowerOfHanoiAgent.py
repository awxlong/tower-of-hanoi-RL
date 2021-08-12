import numpy as np
import random
from TowerOfHanoiEnv import *
import math 
class RLAgent:
    def __init__(self, environment, actions, N0=100, discount_factor=1, _lambda=0.1):
        self.env = environment
        self.actions = actions
        self.N0 = N0
        self.discount_factor = discount_factor
        self._lambda = _lambda
        self.Q = self._init_tenor()
        self.returns_count = self._init_tenor()

    def _init_tenor(self):
        """
        initializing Q(S, A)
        O(6n^3)
        Added + 1 to avoid zero index error or out of bounds
        """
        return np.zeros((self.env.discs + 1, self.env.discs + 1, self.env.discs + 1, self.actions.num_actions))  
        # 2 * 3C2 = 12, 12 possible actions, e.g, move from rod 0 to rod 2 and viceversa

    def _get_alpha(self):
        return 0.5 # for learning rate or stepsize update 

    def _get_epsilon(self):
        return 0.1 # for exploratory moves

    
    def train(self, num_episodes=10):
        for e in range(num_episodes):
            E = self._init_tenor()  
            state = self.env.init_state()
            action = self._policy(state)
            action_idx = self.actions.all_actions.index(action)
            while not state.is_win:
                self.returns_count[len(state.rods[0])][len(state.rods[1])][len(state.rods[2])][action_idx] += 1
                next_state, reward = self.env.step(state, action)
                print(reward.value)
                print(next_state.rods)
                next_action = self._policy(next_state) # there is no action anymore if you've won
                print(next_action)
                
                td_delta = self._td_delta(state, action_idx, next_state, next_action, reward)
                E[len(state.rods[0])][len(state.rods[1])][len(state.rods[2])][action_idx] += 1
                self.Q += self._get_alpha()*td_delta*E
                E = self.discount_factor*self._lambda*E
                state = next_state
                action = next_action
                
            if e % 10 == 0:
                print("\rEpisode {}/{}.".format(e, num_episodes), end="")


        return self.Q

    def _policy(self, state):
        """
        Returns:
        tuple(a, b)
        """
        if state.is_win: return None
        if random.random() < self._get_epsilon():
            return self._get_random_action()
        else:
            epsilon = self._get_epsilon()
            state_actions = np.ones(self.actions.num_actions, dtype=float) * epsilon/self.actions.num_actions
            greedy_action = np.argmax(self.Q[len(state.rods[0])][len(state.rods[1])][len(state.rods[2])])
            state_actions[greedy_action] += (1.0 - epsilon)
            action = np.random.choice(np.arange(len(state_actions)), p=state_actions)
            return self.actions.all_actions[action]

    def _get_random_action(self):
        """
        https://stackoverflow.com/questions/66727094/how-to-randomly-choose-a-tuple-from-a-list-of-tuples
        Returns:
        Tuple(a,b)
        """
        indeces = [i for i in range(self.actions.num_actions)]
        rand_action =  self.actions.all_actions[np.random.choice(indeces)]
        return rand_action
    
    
    def _td_delta(self, s, a, next_s, next_a, r):
        if next_s.is_win:
            return r.value - self.Q[len(s.rods[0])][len(s.rods[1])][len(s.rods[2])][a]
        else:
            next_action_idx = self.actions.all_actions.index(next_a)
            return (r.value + self.discount_factor*self.Q[len(next_s.rods[0])][len(next_s.rods[1])][len(next_s.rods[2])][next_action_idx]) - self.Q[len(s.rods[0])][len(s.rods[1])][len(s.rods[2])][a]


TowerOfHanoiEnv = Environment(5)
LegalActions = Action()
myRLAgent = RLAgent(TowerOfHanoiEnv, LegalActions)
Q = myRLAgent.train(10)
print(Q)