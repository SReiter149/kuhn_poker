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
        log: Dictionary with 'rewards', 'player_ids', 'actions', 'deal' as tensors
    """
    from torch.distributions import Bernoulli
    from tqdm import tqdm
    
    optimizer = AdamW(player.parameters(), lr=config['learning_rate'])
    optimal_bot = OptimalBotP2(device=config['device'])
    
    log = {
        'rewards': [],
        'deal': [],
        'player_ids': [],
        'actions': [],
    }
    
    progress_bar = tqdm(range(config['train_steps']))
    
    for game_id in progress_bar:
        # reset game
        states, terminal = game.reset()
        states = [torch.tensor(s, dtype=torch.float32, device=config['device']) for s in states]
        
        # store P1 info
        p1_log_probs = []
        p1_rewards = []
        turn = 0
        
        while not terminal:
            current_player = turn % 2
            
            if current_player == 0:
                # p1 turn
                probs = player(states[0])
                dist = Bernoulli(probs=probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                p1_log_probs.append(log_prob)
            else:
                # p2 turn
                probs = optimal_bot(states[1])
                dist = Bernoulli(probs=probs)
                action = dist.sample()
            
            # execute action
            states, rewards, terminal = game.step(np.int64(action.item()))
            states = [torch.tensor(s, dtype=torch.float32, device=config['device']) for s in states]
            
            # store P1's reward after each move
            if isinstance(rewards, (list, tuple)):
                p1_reward = rewards[0]
            else:
                p1_reward = rewards
            p1_rewards.append(p1_reward)
            
            turn += 1
        
        # final reward for P1
        if p1_rewards:
            if isinstance(p1_rewards[-1], (list, tuple)):
                final_reward = float(p1_rewards[-1][0])
            elif isinstance(p1_rewards[-1], np.ndarray):
                final_reward = float(p1_rewards[-1].item() if p1_rewards[-1].size == 1 else p1_rewards[-1][0])
            else:
                final_reward = float(p1_rewards[-1])
        else:
            final_reward = 0.0
        
        # log the episode
        log['rewards'].append(torch.tensor(final_reward, dtype=torch.float32, device=config['device']))
        log['player_ids'].append(0)  # always player 0
        log['deal'].append(game.get_deal())
        log['actions'].append(torch.tensor(0, dtype=torch.int64))  # placeholder
        
        # compute loss for P1 (REINFORCE)
        if len(p1_log_probs) > 0:
            if isinstance(final_reward, (list, tuple)):
                reward_value = float(final_reward[0])
            elif isinstance(final_reward, np.ndarray):
                reward_value = float(final_reward.item() if final_reward.size == 1 else final_reward[0])
            else:
                reward_value = float(final_reward)
            
            # stack log probs
            log_probs_tensor = torch.cat([lp.view(-1) for lp in p1_log_probs])
            loss = -(log_probs_tensor.sum() * reward_value)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # progress bar update
        if (game_id + 1) % 1000 == 0:
            recent_rewards = [log['rewards'][i].item() for i in range(max(0, len(log['rewards'])-1000), len(log['rewards']))]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            progress_bar.set_postfix({'avg_reward': f'{avg_reward:.4f}'})
    
    log['rewards'] = torch.stack(log['rewards'])
    log['player_ids'] = torch.tensor(log['player_ids'], device=config['device'])
    log['actions'] = torch.stack(log['actions'])
    # deal is complex, convert via numpy array
    try:
        deal_array = np.array([np.array(d) for d in log['deal']])
        log['deal'] = torch.from_numpy(deal_array).to(config['device'])
    except:
        # if conversion fails, just keep as list
        pass
    
    return log


if __name__ == '__main__':
    try:
        from models import create_kuhn_player
        
        config = {
            'device': 'cpu',
            'replay_buffer_capacity': 1_000,
            'minibatch_size': 10,
            'warmup_steps': 100,
            'train_steps': 10_000,
            'log_window': 100,
            'batch_size': 32,
            'learning_rate': 0.02
        }

        kuhn = KuhnPoker()

        # make players
        num_players = 4
        players = [create_kuhn_player(config['device']) for i in range(num_players)]

        # train
        log = train(config, players)

        print("\nTraining complete!")
        print(f"Final rewards shape: {log['rewards'].shape}")

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
