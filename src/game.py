from torch import randperm
# import torch
import pdb
import numpy as np

import pdb
from rich.console import Console

class KuhnPoker():
    def __init__(self,):
        self.reset()

    def deal(self):
        card1, card2 = np.random.choice(3, size = 2, replace = False)
        self.cards[0][card1] = 1
        self.cards[1][card2] = 1

    def get_state(self):
        cards = self.cards.flatten()
        history = self.history.flatten()

        state = np.concatenate([cards, history])

        return state


    def reset(self):
        """
        returns: 
        - state
        - turn
        """
        self.turn = 0
        self.history = np.array([[0,0], [0,0], [0,0]], dtype = np.float32) # action1, action2, action3, [1,0] is pass/fold [0,1] is bet/call and [0,0] is no action
        self.cards = np.array([[0,0,0], [0,0,0]]) # one-hot encoding of cards
        
        self.deal()
        state = self.get_state()
        self.terminal = False

        return state, self.terminal
    
    def check_history(self):
        """
        returns:
        - terminal (bool)
        - winner (int): 0 is showdonw, 1 is player 1 won, 2 is player 2 won
        - pot (int)
        """
        if (self.history[:2] == np.array([[1,0], [1,0]], dtype = np.float32)).all():
            return (True, 0, 1)
        if (self.history == np.array([[1,0], [0,1], [1,0]], dtype = np.float32)).all():
            return (True, 2, 1)
        if (self.history == np.array([[1,0], [0,1], [0,1]], dtype = np.float32)).all():
            return (True, 0, 2)
        if (self.history[:2] == np.array([[0,1], [1,0]], dtype = np.float32)).all():
            return (True, 1, 1)
        if (self.history[:2] == np.array([[0,1], [0,1]], dtype = np.float32)).all():
            return (True, 0, 1)

        return (False, None, None)
    
    def showdown(self):
        card1, card2 = self.cards
        if np.where(card1 == 1) > np.where(card2 == 1):
            return 1
        elif np.where(card1 == 1) < np.where(card2 == 1):
            return 2
        else:
            raise ValueError("logic error, neither card 1 nor card 2 is bigger than the other")
    
    def step(self,action):
        """
        action:
        - 0 is fold/pass
        - 1 is bet/call
        """
        assert action.dtype == np.int64

        self.history[self.turn][action] = 1
        terminal, winner, pot = self.check_history()
        state = self.get_state()

        if terminal:
            if winner == 0:
                winner = self.showdown()

            if winner == 1:
                reward = np.array([pot, -pot], dtype = np.float32)
            elif winner == 2:
                reward = np.array([-pot, pot], dtype = np.float32)
            return state, reward, terminal

        self.turn += 1
        reward = np.array([0,0], dtype = np.float32)
        return state, reward, terminal

if __name__ == "__main__":
    try:   
        kuhn = KuhnPoker()
        state, terminal = kuhn.reset()
        state, reward, terminal = kuhn.step(1)
        state, reward, terminal = kuhn.step(1)
        # state, reward, terminal = kuhn.step(0)
        

        print(state)
        print(reward)
        print(terminal)

    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()




    

