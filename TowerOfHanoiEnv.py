import numpy as np 
from enum import Enum

class Environment:
    '''
    Intuitive environment for Tower of Hanoi in which
    the RL agent will learn to play
    '''
    def __init__(self, discs):
        self.state = State(discs)
        self.discs = discs

    def init_state(self):
        self.state.rods[0] = [i for i in reversed(range(self.discs))]
        self.state.is_win = False
        self.state.num_steps = 0
        return self.state

    def step(self, state, action):
        """
        Arguments:
        state: gives access to current states of three rods
        actions: tuple of size 2 (a, b), where a represents
        the rod from which a disk is moved, b the rod moved to.
        Returns:
        next_state
        reward: depending on whether there was an illegal move, 
        reached the goal, or otherwise made a normal step
        """
        next_state = state.rods[action[1]].append(state.rods[action[0]].pop())  # append disc to rod 1 popped from rod 0
        next_state.num_steps += 1  # to check that agent learns to solve TOH in 2^N - 1 steps
        if self.is_illegal(next_state.rods[action[1]]):
            next_state = next_state.rods[action[0]].append(state.rods[action[1]].pop())  # reverse the illegal move
            return next_state, Reward.REVERSE
        elif self.is_victory(next_state):
            next_state.is_win = True
            return next_state, Reward.GOAL

        return next_state, Reward.STEP
        
    def is_illegal(self, rod):
        if len(rod) == 1:
            return False
        else:
            if rod[len(rod)] > rod[len(rod) - 1]:
                return True 
    def is_victory(self, state):
        return state.rods[2] == [i for i in reversed(range(self.discs))]
    


# class Action(Enum):
#     POP = 1

class Reward (Enum):
    STEP = -1
    GOAL = 10
    REVERSE = -50

class State:
    def __init__(self, discs, is_win = False):
        self.rods = {0: [], 1: [], 2: []}
        # self.rod_a = self.rods[0]
        # self.rod_b = self.rods[1]
        # self.rod_c = self.rods[2]
        self.is_win = is_win
        self.num_steps = 0

