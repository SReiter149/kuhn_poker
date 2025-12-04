import torch
from torch.nn import Linear, Sequential, ReLU
from torch.distributions import Categorical
from torch.optim import AdamW
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
import gymnasium as gym

# from model import Player1, Player2
from game import KuhnPoker
from util import get_seed, get_mlp
from analyze import analyze_strategy

from matplotlib import pyplot as plt

import pdb
from rich.console import Console

def play_game(player1,player2, game):
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
        'player1_ids': [],
        'player2_ids': [],
        'actions': [],
    }
    progress_bar = tqdm(range(config['train_steps']))
    loss = [torch.tensor(0.0, device=config['device']) for _ in players]

    player1_id, player2_id = torch.randperm(len(players))[:2]

    for game_id in progress_bar:
        # play a game
        memory, deal = play_game(players[player1_id], players[player2_id], kuhn)
        log['deal'].append(deal)
        log['actions'].append((memory[0]['actions'],memory[1]['actions']))
        log['rewards'].append(memory[0]['rewards'][-1])
        log['player1_ids'].append(player1_id)
        log['player2_ids'].append(player2_id)
            
        # back propogation
        for player_num, player_id in enumerate([player1_id, player2_id]):
            # get undiscounted rewards (undiscounted since number of moves shouldn't matter)
            if len(memory[player_num]['rewards']) > 1:
                rewards = torch.tensor([sum(memory[player_num]['rewards'][move:]) for move in range(len(memory[player_num]['rewards']))], device= config['device'], dtype = torch.float32)
            else:
                rewards = torch.tensor(memory[player_num]['rewards'][0], device=config['device'], dtype=torch.float32)
                
            # loss and backprop
            # CHECK MAKE SURE THIS LOSS FUNCTION IS OPTIMAL
            game_loss = -1 * ((memory[player_num]['log_probs'] * rewards).sum())
            loss[player_num] = loss[player_num] + game_loss

            if (game_id + 1) % config['batch_size'] == 0:
                loss[player_num].backward()
                optimizers[player_id].step()

                optimizers[player_id].zero_grad()
                loss = [torch.tensor(0.0, device=config['device']) for _ in range(2)]

                player1_id, player2_id = torch.randperm(len(players))[:2]
    for key in ('player1_ids', 'player2_ids', "rewards"):
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
        player1 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player2 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        # player3 = Sequential(
        #     Linear(9, 12),
        #     ReLU(),
        #     Linear(12, 10),
        #     ReLU(),
        #     Linear(10, 6),
        #     ReLU(),
        #     Linear(6,2)
        # )
        # player4 = Sequential(
        #     Linear(9, 12),
        #     ReLU(),
        #     Linear(12, 10),
        #     ReLU(),
        #     Linear(10, 6),
        #     ReLU(),
        #     Linear(6,2)
        # )

        players = [
            player1,
            player2,
            # player3,
            # player4
        ]

        # train
        log = train(config, players)
        # pdb.set_trace()

        # print analysis
        analyze_strategy(players[0])
        
        # visualize
        fig, ax = plt.subplots(2,2)

        as_player1 = log['player1_ids'] == 0
        as_player2 = log['player2_ids'] == 0

        rewards = log['rewards'][as_player1 | as_player2]
        rewards[as_player2] *= -1

        # plot average rewards
        kernel = torch.ones(config['log_window']) / config['log_window']
        rewards = torch.tensor(rewards, dtype=torch.float32)
        average_score = torch.conv1d(
            rewards.view(1,1,-1),
            kernel.view(1,1,-1)
        ).view(-1)
        ax[0,0].plot(average_score, label = 'player1 rewards')
        ax[0,0].axhline(y=float(1/18), color='r', linestyle='--', label='Nash (P1 = 1/18)')
        
        # ax[0,0].set_yscale('log')

        ax[0,1].plot(torch.cumsum(rewards, dim = 0))
        plt.legend()
        plt.show()

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
