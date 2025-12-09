import torch
from torch.nn import Linear, Sequential, Sigmoid
from torch.distributions import Categorical, Bernoulli
from torch.optim import AdamW
torch.autograd.set_detect_anomaly(True)

import numpy as np

from tqdm import tqdm
import gymnasium as gym

# from model import Player1, Player2
from game import KuhnPoker
from util import get_seed, get_mlp
from analyze import analyze_strategy, plot_training

from matplotlib import pyplot as plt

import pdb
from rich.console import Console

def play_game(player1, game, player2 = None):
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
    if player2 == None:
        players = [player1]
    else:
        players = [player1, player2]  
    num_players = len(players)
    memory = [
        {
            "rewards": [],
            'actions': [],
            "observations": [],
            "log_probs": [],
        } for i in range(len(players))
    ]
    turn = 0
    terminal = False
    states,terminal = game.reset()

    states = [
        torch.asarray(state, device = config['device'], dtype = torch.float32) 
        for state in states
    ]

    while not terminal:
        player = turn % num_players
        # get active players probability distribution and action

        probs = players[player](states[player])
        # pdb.set_trace()

        dist = Bernoulli(probs = probs)
        action = dist.sample()                         
        log_prob = dist.log_prob(action)

        # play the move
        states, reward, terminal = game.step(np.int64(action))

        # convert to torch
        states = [
            torch.asarray(state, device = config['device'], dtype = torch.float32) 
            for state in states
        ]
        reward = torch.asarray(reward, device = config['device'], dtype = torch.float32)
        terminal = torch.asarray(terminal, device = config['device'], dtype = torch.bool)

        # update last players buffer based on current players move
        if turn >= 1:
            assert last_log_prob.shape == torch.Size([1])
            memory[(player + 1) % num_players]['log_probs'].append(last_log_prob)
            memory[(player + 1) % num_players]['observations'].append(last_states[(player + 1) % num_players])
            memory[(player + 1) % num_players]['rewards'].append(reward[(player + 1) % num_players])
            memory[(player + 1) % num_players]['actions'].append(last_action)

        # set up for next turn
        last_log_prob = log_prob
        last_states = states
        last_action = action
        turn += 1

    # save the last turn for the active player
    memory[player]['log_probs'].append(last_log_prob)
    memory[player]['observations'].append(last_states[player])
    memory[player]['rewards'].append(reward[player])
    memory[player]['actions'].append(last_action)
    
    # stack all lists
    for player in range(num_players):
        for key in memory[player].keys():
            memory[player][key] = torch.stack(memory[player][key])
    
    if len(memory) == 1:
        return memory[0], game.get_deal()
    return memory, game.get_deal()


def train(config, players):
    """
    train the two models
    train (config, [player1, player2])

    returns:
    - log (dict):
        - p1_rewards (list): the rewards for the first player
        - p1_score (list): the sum of the rewards up to that point
    """

    # set up
    optimizers = [AdamW(players[i].parameters(), lr = config['learning_rate']) for i in range(len(players))]
    log = {
        'rewards': [],
        'deal': [],
        'player_ids': [],
        'actions': [],
    }
    progress_bar = tqdm(range(config['train_steps']))
    loss = torch.tensor(0.0, device=config['device'])
    player_id = 0

    for game_id in progress_bar:
            # play a game
            memory, deal = play_game(players[player_id], kuhn)
            log['deal'].append(deal)
            log['actions'].append((memory['actions']))
            log['rewards'].append(memory['rewards'][-1])
            log['player_ids'].append(player_id)
            
            if len(memory['rewards']) > 1:
                rewards = torch.tensor([sum(memory['rewards'][move:]) for move in range(len(memory['rewards']))], device= config['device'], dtype = torch.float32)
            else:
                rewards = torch.tensor(memory['rewards'][0], device=config['device'], dtype=torch.float32)
                
            # loss and backprop
            # CHECK MAKE SURE THIS LOSS FUNCTION IS OPTIMAL
            game_loss = -1 * ((memory['log_probs'] * rewards).sum())
            loss = loss + game_loss

            if (game_id + 1) % config['batch_size'] == 0:
                loss.backward()
                optimizers[player_id].step()

                optimizers[player_id].zero_grad()
                loss = torch.tensor(0.0, device=config['device'])

                player_id += 1
                player_id %= num_players

    for key in ('player_ids', "rewards"):
        log[key] = torch.tensor(log[key])
    return log

if __name__ == '__main__':
    try:
        config = {
            'device': 'cpu',
            'replay_buffer_capacity': 1_000,
            'minibatch_size': 10,
            'warmup_steps': 100,
            'train_steps': 100_000,
            'log_window': 1_000,
            'batch_size': 128,
            'learning_rate': 0.02
        }

        kuhn = KuhnPoker()

        # make players
        num_players = 4
        players = [
            Sequential(
                Linear(12, 1),
                Sigmoid()
            ) for i in range(num_players)
        ]

        # train
        log = train(config, players)

        # analyze
        for player in players:
            print("---------")
            analyze_strategy(player)

        # plot
        plot_training(config, players, log)

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
