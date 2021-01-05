import numpy as np
import random
from TowerOfHanoiEnv import *

class RLAgent:
    def __init__(self, environment, N0=100, discount_factor=1, _lambda=0.1):
        self.env = environment
        self.N0 = N0
        self.discount_factor = discount_factor
        self._lambda = _lambda
        self.Q = self._init_tenor()

    def _init_tenor(self):
        "initializing Q(S, A)?"
        return np.zeros((self.env.discs, self.env.discs, self.env.discs, 12))  # 2 * 3C2 = 12, 12 possible actions, e.g, move from rod 0 to rod 2 and viceversa

    def _get_alpha(self):

        return 0.5 # for learning rate or stepsize update 


    def _get_epsilon(self, s):
        
        return 0.01 # for exploratory moves

    
    def train(self, num_episodes=10):
        for e in range(num_episodes):
            E = self._init_tenor()
            s = self.env.init_state()
            a = self._policy(s)
            while not s.is_won:
                self.returns_count[s.row][s.column][a] += 1
                next_s, r = self.env.step(s, Action(a))
                next_a = self._policy(next_s)
                td_delta = self._td_delta(s, a, next_s, next_a, r)
                E[s.row][s.column][a] += 1
                self.Q += self._get_alpha(s, a)*td_delta*E
                E = self.discount_factor*self._lambda*E
                s = next_s
                a = next_a

            if e % 10 == 0:
                print("\rEpisode {}/{}.".format(e, num_episodes), end="")


        return self.Q

    def _policy(self, s):
        if s.is_won: return None
        if random.random() < self._get_epsilon(s):
            return self._get_random_action()
        else:
            return self._get_greedy_action(s)

    def _get_epsilon(self, s):
        return 0.05

    def _get_random_action(self):
        rods = [0, 1, 2]
        rand_action = np.random.choice(rods,2, replace=False)
        return rand_action

    def _get_greedy_action(self, s):
        return np.argmax([self._get_Q(self._get_phi(s, a.value)) for a in Action])

    def _td_delta(self, s, a, next_s, next_a, r):
        if next_s.is_won:
            return r.value - self.Q[s.row][s.column][a]
        else:
            return (r.value + self.discount_factor*self.Q[next_s.row][next_s.column][next_a]) - self.Q[s.row][s.column][a]
