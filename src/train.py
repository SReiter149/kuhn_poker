import torch
from torch.nn import Linear, Sequential
from torch.distributions import Categorical
from torch.optim import AdamW

from tqdm import tqdm
import gymnasium as gym

# from model import Player1, Player2
from game import KuhnPoker
from util import get_seed, get_mlp

from matplotlib import pyplot as plt

import pdb
from rich.console import Console

def play_game(players, game):
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
        - probabilities (list): the probability that the chosen move should have been chosen
    '''

    # set up    
    memory = [
        {
            "rewards": [],
            "observations": [],
            "probabilities": [],
        } for i in range(len(players))
    ]
    turn = 0
    terminal = False
    state,terminal = game.reset()
    state = torch.asarray(state, device = config['device'], dtype = torch.float32)

    while not terminal:
        player = turn % 2

        # get active players probability distribution and action
        action_probabilities = players[player](state)
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
            memory[(player + 1) % 2]['probabilities'].append(last_log_prob)
            memory[(player + 1) % 2]['observations'].append(last_state)
            memory[(player + 1) % 2]['rewards'].append(reward[(player + 1) % 2])

        # set up for next turn
        last_log_prob = log_prob
        last_state = state
        turn += 1

    # save the last turn for the active player
    memory[player]['probabilities'].append(last_log_prob)
    memory[player]['observations'].append(last_state)
    memory[player]['rewards'].append(reward[player])
    
    # stack all lists
    for player in range(2):
        for key in memory[player].keys():
            memory[player][key] = torch.stack(memory[player][key])
    return memory


def train(config, players):
    """
    train the two models

    returns:
    - log (dict):
        - player1_rewards (list): the rewards for the first player
        - player1_score (list): the sum of the rewards up to that point
    """

    # set up
    optimizers = [AdamW(players[i].parameters()) for i in range(len(players))]
    log = {
        'player1_rewards': [],
        'player1_score': []
    }
    progress_bar = tqdm(range(config['train_steps']))

    for game_id in progress_bar:
        # play a game
        memory = play_game(players, kuhn)

        # save results to log
        log['player1_rewards'].append(memory[0]['rewards'][-1])
        if game_id >= 1:
            log['player1_score'].append(log['player1_score'][-1] + log['player1_rewards'][-1])
        else:
            log['player1_score'].append(log['player1_rewards'][-1])
            
        # back propogation
        for player in range(len(players)):
            # get undiscounted rewards (undiscounted since number of moves shouldn't matter)
            if len(memory[player]['rewards']) > 1:
                rewards = torch.tensor([sum(memory[player]['rewards'][move:]) for move in range(len(memory[player]['rewards']))], device= config['device'], dtype = torch.float32)
            else:
                rewards = memory[player]['rewards'][0]

            # loss and backprop
            # CHECK MAKE SURE THIS LOSS FUNCTION IS OPTIMAL
            optimizers[player].zero_grad()
            loss = -1 * ((memory[player]['probabilities'] * rewards).sum())
            loss.backward()
            optimizers[player].step()
    
    return log

if __name__ == '__main__':
    try:
        config = {
            'device': 'cpu',
            'replay_buffer_capacity': 1_000,
            'minibatch_size': 10,
            'warmup_steps': 100,
            'train_steps': 100_000
        }

        kuhn = KuhnPoker()

        # make players
        player1 = Sequential(
            Linear(12, 10),
            Linear(10, 10),
            Linear(10, 6),
            Linear(6,2)
        )
        player2 = Sequential(
            Linear(12, 10),
            Linear(10, 10),
            Linear(10, 6),
            Linear(6,2)
        )

        players = [
            player1,
            player2
        ]

        # train
        log = train(config, players)
        
        # visualize
        plt.plot(log['player1_score'])
        plt.show()

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
