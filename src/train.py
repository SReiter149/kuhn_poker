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
from models import OptimalBotP2
from analyze import distance_from_optimal

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
        - reward (list): the reward after the next players move at each move
        - observations (list): the observations for each player
        - log_probs (list): the probability that the chosen move should have been chosen
    '''

    # set up  
    if player2 == None:
        players = [player1]
    else:
        players = [player1, player2]  
    num_players = len(players)
    device = player1[0].weight.device
    memories = [
        {
            "reward": None,
            'actions': [],
            "observations": [],
            "log_probs": [],
        } for i in range(len(players))
    ]
    turn = 0
    terminal = False
    states,terminal = game.reset()

    states = [
        torch.asarray(state, device = device, dtype = torch.float32) 
        for state in states
    ]

    while not terminal:
        player = turn % num_players

        # save initial observation
        memories[player]['observations'].append(states[player])

        # get active players probability distribution and action
        probs = players[player](states[player])
        dist = Bernoulli(probs = probs)
        action = dist.sample()                         
        log_prob = dist.log_prob(action)

        # play the move
        states, reward, terminal = game.step(np.int64(action))
        

        # convert to torch
        states = [
            torch.asarray(state, device = device, dtype = torch.float32) 
            for state in states
        ]

        # save reward and terminal
        reward = torch.asarray(reward, device = device, dtype = torch.float32)
        terminal = torch.asarray(terminal, device = device, dtype = torch.bool)
        
        memories[player]['log_probs'].append(log_prob)
        memories[player]['actions'].append(action)

        turn += 1
        
    for player in range(num_players):
        memories[player]['reward'] = reward[player]        
    
    # stack all lists
    for player in range(num_players):
        for key in ['actions', 'observations', 'log_probs']:
            memories[player][key] = torch.stack(memories[player][key])
    
    if len(memories) == 1:
        memories[0]['reward'] = reward
        return memories[0], game.get_deal()
    return memories, game.get_deal()

def train_self_play(config, player, game, gamma = 0.1):
    """
    note: baseline = (1-gamma) * baseline + gamma * new_reward

    returns logs, player
    """

    optimizer = AdamW(player.parameters(), lr=config['learning_rate'])
    rewards = torch.tensor([0,0],dtype = torch.float32, device = config['device'])
    baselines = torch.tensor([-2.0, -2.0], dtype = torch.float32, device = config['device'])
    loss = 0.0
    
    log = {
        'reward': [],
        'deals': [],
        'player_ids': [],
        'actions': [],
        'distances': [],
    }
    
    progress_bar = tqdm(range(config['train_steps']))
    

    for game_id in progress_bar:

        memories, deal = play_game(
            player1 = player, 
            player2 = player,
            game = game
        )
         # isolate first players memory

        # log the episode
        
        log['reward'].append(memories[0]['reward'].detach().clone().to(torch.float32).to(config['device']))
        log['player_ids'].append(0)  # always player 0
        log['deals'].append(torch.tensor(deal))
        log['actions'].append(([memories[0]['actions']], memories[1]['actions']))  # placeholder
        
        # compute loss for P1 (REINFORCE)
        for i in range(2):
            final_reward = memories[i]['reward']
            game_reward = -1 * ((memories[i]['log_probs'] * final_reward).sum())

            game_advantage = final_reward - baselines[i]
            game_loss = -1 * ((memories[i]['log_probs'] * game_advantage).sum())

            rewards[i] = rewards[i] + game_reward
            loss = loss + game_loss
            # pdb.set_trace()
        

        if (game_id + 1) % config['batch_size'] == 0:
            rewards = rewards / (2 * config['batch_size'])
            loss = loss / (2 * config['batch_size'])
            baselines = 0.99 * baselines + 0.01 * rewards

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = 0
            rewards = torch.tensor([0,0],dtype = torch.float32, device = config['device'])
            baselines = torch.tensor([-2.0, -2.0], dtype = torch.float32, device = config['device'])

            distance1, distance2 = distance_from_optimal(player)
            log['distances'].append(distance1 + distance2)
        
        # progress bar update
        if (game_id + 1) % 100 == 0:
            recent_reward = [log['reward'][i].item() for i in range(max(0, len(log['reward'])-1000), len(log['reward']))]
            avg_reward = np.mean(recent_reward) if recent_reward else 0.0
            progress_bar.set_postfix({'avg_reward': f'{avg_reward:.4f}'})
    
    log['reward'] = torch.stack(log['reward'])
    log['player_ids'] = torch.tensor(log['player_ids'], device=config['device'])
    # log['actions'] = torch.stack(log['actions'])
    log['deals'] = torch.stack(log['deals'])
    
    return player, log


def train(config, players):
    """
    train the two models
    train (config, [player1, player2])

    returns:
    - log (dict):
        - p1_reward (list): the reward for the first player
        - p1_score (list): the sum of the reward up to that point
    """

    # set up
    optimizers = [AdamW(players[i].parameters(), lr = config['learning_rate']) for i in range(len(players))]
    baselines = [-2 for _ in range(len(players))]
    log = {
        'reward': [],
        'deal': [],
        'player_ids': [],
        'actions': [],
    }
    progress_bar = tqdm(range(config['train_steps']))
    reward = torch.tensor(0.0, device=config['device'])
    player_id = 0

    for game_id in progress_bar:
            # play a game
            memories, deal = play_game(players[player_id], kuhn)
            log['deal'].append(deal)
            log['actions'].append((memories['actions']))
            log['reward'].append(memories['reward'])
            log['player_ids'].append(player_id)
            
            if len(memories['reward']) > 1:
                reward = torch.tensor([sum(memories['reward'][move:]) for move in range(len(memories['reward']))], device= config['device'], dtype = torch.float32)
            else:
                reward = torch.tensor(memories['reward'][0], device=config['device'], dtype=torch.float32)
                
            # loss and backprop
            # CHECK MAKE SURE THIS LOSS FUNCTION IS OPTIMAL
            game_reward = -1 * ((memories['log_probs'] * reward).sum())
            reward = reward + game_reward

            if (game_id + 1) % config['batch_size'] == 0:
                baselines[player_id]
                reward.backward()
                optimizers[player_id].step()

                optimizers[player_id].zero_grad()
                reward = torch.tensor(0.0, device=config['device'])

                player_id += 1
                player_id %= num_players

    for key in ('player_ids', "reward"):
        log[key] = torch.tensor(log[key])
    return player, log


def train_vs_optimal_bot(config, player, game):
    """
    Train a single player against the optimal Nash equilibrium bot.
    This allows P1 to converge to Nash equilibrium value of -1/18.
    
    Args:
        config: Training configuration dict with keys:
            - device: 'cpu' or 'cuda'
            - learning_rate: float
            - train_steps: int
        player: P1 neural network model
        game: KuhnPoker environment
        
    Returns:
        log: Dictionary with 'reward', 'player_ids', 'actions', 'deal' as tensors
    """
    from torch.distributions import Bernoulli
    from tqdm import tqdm
    
    optimizer = AdamW(player.parameters(), lr=config['learning_rate'])
    loss = torch.tensor(0.0, device=config['device'])

    optimal_bot = OptimalBotP2(device=config['device'])
    
    log = {
        'reward': [],
        'deals': [],
        'player_ids': [],
        'actions': [],
        'distances': [],
    }
    
    progress_bar = tqdm(range(config['train_steps']))
    

    for game_id in progress_bar:

        memories, deal = play_game(
            player1 = player, 
            player2 = optimal_bot,
            game = game
        )
        memory = memories[0] # isolate first players memory

        # log the episode
        
        log['reward'].append(memory['reward'].detach().clone().to(torch.float32).to(config['device']))
        log['player_ids'].append(0)  # always player 0
        log['deals'].append(torch.tensor(deal))
        log['actions'].append([memory['actions'], memories[1]['actions']])  # placeholder
        
        # compute loss for P1 (REINFORCE)
        final_reward = memory['reward']
        # pdb.set_trace()

        game_loss = -1 * ((memory['log_probs'] * final_reward).sum())
        # pdb.set_trace()
        # game_loss = -1 * final_reward
        loss = loss + game_loss

        if (game_id + 1) % config['batch_size'] == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=config['device'])
            distance = distance_from_optimal(player)[0]
            log['distances'].append(distance)
        
        # progress bar update
        if (game_id + 1) % 100 == 0:
            recent_reward = [log['reward'][i].item() for i in range(max(0, len(log['reward'])-1000), len(log['reward']))]
            avg_reward = np.mean(recent_reward) if recent_reward else 0.0
            progress_bar.set_postfix({'avg_reward': f'{avg_reward:.4f}'})
    
    log['reward'] = torch.stack(log['reward'])
    log['player_ids'] = torch.tensor(log['player_ids'], device=config['device'])
    # log['actions'] = torch.stack(log['actions'])
    log['deals'] = torch.stack(log['deals'])
    
    return player, log


if __name__ == '__main__':
    try:
        from models import create_kuhn_player
        from analyze import distance_from_optimal
        
        config = {
            'device': 'cpu',
            'replay_buffer_capacity': 1_000,
            'minibatch_size': 10,
            'warmup_steps': 100,
            'train_steps': 100_000,
            'log_window': 100,
            'batch_size': 32,
            'learning_rate': 0.01
        }

        kuhn = KuhnPoker()

        # # test train vs optimal bot
        # player = create_kuhn_player(config['device'])
        # player, log = train_vs_optimal_bot(config, player, kuhn)
        # print(f"distance: {distance_from_optimal(player)[0]}")
        # print(f"probabilities: {torch.sigmoid(player[0].weight)} (note remember to ignore the probs for when its player2)")
        # plt.plot(log['distances'])
        # plt.show()
        # pdb.set_trace()

        player = create_kuhn_player(config['device'])
        player, log = train_self_play(config, player, kuhn)
        print(f"distance: {distance_from_optimal(player)[0]}")
        print(f"probabilities: {torch.sigmoid(player[0].weight)} (note remember to ignore the probs for when its player2)")
        plt.plot(log['distances'])
        plt.show()
        pdb.set_trace()

        # test self play
        # make players
        # num_players = 4
        # players = [create_kuhn_player(config['device']) for i in range(num_players)]

        # # train
        # log = train(config, players)

        # print("\nTraining complete!")
        # print(f"Final reward shape: {log['reward'].shape}")

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
