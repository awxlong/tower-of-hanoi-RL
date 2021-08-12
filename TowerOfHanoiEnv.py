import numpy as np 
from enum import Enum
from itertools import permutations
from copy import copy

# reference: https://github.com/clarisli/RL-Easy21
# 
class Environment:
    '''
    Intuitive environment for Tower of Hanoi in which
    the RL agent will learn to play
    '''
    def __init__(self, discs):
        """
        The 0 rod means there isn't any rod. I added it in order to 
        take into account the possibility of anction popping from an empty list.
        """
        self.rods = {0:[], 1:[], 2:[]}
        self.state = State(self.rods)
        self.discs = discs

    def init_state(self):
        self.state.rods = {0:[], 1:[], 2:[]}
        self.state.rods[0] += [i for i in reversed(range(self.discs))] 
        self.state.is_win = False
        self.state.num_steps = 0
        return self.state

    def moveFrom(self, myRod):
        if len(myRod) == 0:
            return []
        else:
            return [myRod.pop()]

    def step(self, state, action):
        """
        KEY FUNCTION
        Arguments:
        state: gives access to current states of three rods
        action: tuple of size 2 (a, b), where a represents
        the rod from which a disk is moved, b the rod moved to.
        Returns:
        next_state
        reward: depending on whether there was an illegal move, 
        reached the goal, or otherwise made a normal step
        """
        next_state = copy(state)
        next_state.rods[action[1]] += self.moveFrom(next_state.rods[action[0]])
        #next_state.rods[action[1]].append(next_state.rods[action[0]].pop())  # append disc to rod 1 popped from rod 0
        next_state.num_steps += 1  # to check that agent learns to solve TOH in 2^N - 1 steps
        if self.is_illegal(next_state.rods[action[1]]):
            next_state.rods[action[0]].append(state.rods[action[1]].pop())  # reverse the illegal move
            return next_state, Reward.REVERSE
        elif self.is_victory(next_state):
            next_state.is_win = True
            return next_state, Reward.GOAL

        return next_state, Reward.STEP
        
    def is_illegal(self, rod):
        """
        For example, 0: [5, 4, 3, 2, 0, 1] is illegal 

        """
        if (len(rod) == 1) or (len(rod) == 0):
            return False
        elif rod[-1] > rod[-2]:
            return True 
    def is_victory(self, state):
        return state.rods[2] == [i for i in reversed(range(self.discs))]

class Reward (Enum):
    STEP = -1
    GOAL = 10
    REVERSE = -50

class State:
    def __init__(self, rods, is_win = False):
        self.rods = rods
        self.is_win = is_win
        self.num_steps = 0  # to check if agent completed in 2**discs - 1 steps

class Action:
    """
    A field of this class gets all the possible moves that 
    the rod can move from, to.
    
    """
    def __init__(self):
        self.all_actions = list(permutations([0, 1, 2], 2))
        # all actions are [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        self.num_actions = len(self.all_actions)
        self.enum = list(enumerate(self.all_actions)) # for filling the Q(S,A) matrix