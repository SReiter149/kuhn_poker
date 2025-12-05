from torch import randperm
# import torch
import pdb
import numpy as np
import torch
from torch.distributions import Categorical

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
        cards = self.cards.flatten() # (6,)
        history = self.history.flatten() # (6,)

        state = np.concatenate([cards, history]) # (12,)

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
        action:
        - [1,0] is fold/pass
        - [0,1] is bet/call
        returns:
        - terminal (bool)
        - winner (int): 0 is showdonw, 1 is player 1 won, 2 is player 2 won
        - pot (int)
        """
        if (self.history[:2] == np.array([[1,0], [1,0]], dtype = np.float32)).all():
            return (True, 0, 1)
        if (self.history[:2] == np.array([[0,1], [1,0]], dtype = np.float32)).all():
            return (True, 1, 1)
        if (self.history[:2] == np.array([[0,1], [0,1]], dtype = np.float32)).all():
            return (True, 0, 2)
        if (self.history == np.array([[1,0], [0,1], [1,0]], dtype = np.float32)).all():
            return (True, 2, 1)
        if (self.history == np.array([[1,0], [0,1], [0,1]], dtype = np.float32)).all():
            return (True, 0, 2)
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

def play_game(player1,player2, game, config):
    '''
    play a single game given the players

    ---
    arguments:
    - players (list of models): the players
    - game (env): the enviornment to play in

    ---
    returns:
    - memory (list of dict): a set of memories for each player
        - rewards (list): the rewards after the next players move at each move
        - observations (list): the observations for each player
        - log_probs (list): the probability that the chosen move should have been chosen
    '''

    # set up  
    players = [player1, player2]  
    memory = [
        {
            "rewards": [],
            'actions': [],
            "observations": [],
            "log_probs": [],
        } for i in range(2)
    ]
    turn = 0
    terminal = False
    state,terminal = game.reset()
    state = torch.asarray(state, device = config['device'], dtype = torch.float32)

    while not terminal:
        player = turn % 2
        # pdb.set_trace()
        if player == 0:
            masked_state = torch.cat((state[:3], state[6:]), dim = 0)
        else: # player == 1
            masked_state = state[3:]


        # get active players probability distribution and action
        action_probabilities = players[player](masked_state)
        dist = Categorical(logits=action_probabilities)
        action = dist.sample()                         
        log_prob = dist.log_prob(action).squeeze(0)

        # play the move
        state, reward, terminal = game.step(action.numpy())

        # convert to torch
        state = torch.asarray(state, device = config['device'], dtype = torch.float32)
        reward = torch.asarray(reward, device = config['device'], dtype = torch.float32)
        terminal = torch.asarray(terminal, device = config['device'], dtype = torch.bool)

        # update last players buffer based on current players move
        if turn >= 1:
            assert len(last_log_prob.shape) == 0 
            memory[(player + 1) % 2]['log_probs'].append(last_log_prob)
            memory[(player + 1) % 2]['observations'].append(last_state)
            memory[(player + 1) % 2]['rewards'].append(reward[(player + 1) % 2])
            memory[(player + 1) % 2]['actions'].append(last_action)

        # set up for next turn
        last_log_prob = log_prob
        last_state = state
        last_action = action
        turn += 1

    # save the last turn for the active player
    memory[player]['log_probs'].append(last_log_prob)
    memory[player]['observations'].append(last_state)
    memory[player]['rewards'].append(reward[player])
    memory[player]['actions'].append(last_action)
    
    # stack all lists
    for player in range(2):
        for key in memory[player].keys():
            memory[player][key] = torch.stack(memory[player][key])
    return memory, last_state[:6]

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




    

