import torch
from torch.nn import Linear, Sequential, ReLU
from torch.distributions import Categorical
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
import gymnasium as gym

# from model import Player1, Player2
from game import KuhnPoker, play_game
from util import get_seed, get_mlp, AdamW
from analyze import analyze_strategy
from population_based_training import pbt_step, pbt_init

from matplotlib import pyplot as plt

import pdb
from rich.console import Console

def train(config, players, player_configs):
    """
    train the two models
    train (config, [player1, player2])

    returns:
    - log (dict):
        - p1_rewards (list): the rewards for the first player
        - p1_score (list): the sum of the rewards up to that point
    """
    assert config['pbt_frequency'] % config['batch_size'] == 0
    # set up
    player_configs = pbt_init(player_configs)

    optimizers = [AdamW(players[i].parameters(), player_configs[i]) for i in range(len(players))]
    log = {
        'rewards': [[] for _ in range(len(players))],
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
        memory, deal = play_game(players[player1_id], players[player2_id], kuhn, config)
        log['deal'].append(deal)
        log['actions'].append((memory[0]['actions'],memory[1]['actions']))
        log['rewards'][player1_id].append(memory[0]['rewards'][-1])
        log['rewards'][player2_id].append(memory[1]['rewards'][-1])
        log['player1_ids'].append(player1_id)
        log['player2_ids'].append(player2_id)
            
        # accumulate loss
        for player_num, player_id in enumerate([player1_id, player2_id]):
            # get undiscounted rewards (undiscounted since number of moves shouldn't matter)
            if len(memory[player_num]['rewards']) > 1:
                rewards = torch.sum(memory[player_num]['rewards'], dim = 0)
            else:
                rewards = memory[player_num]['rewards'][0]
                
            # loss
            game_loss = -1 * ((memory[player_num]['log_probs'] * rewards).sum())
            loss[player_num] = loss[player_num] + game_loss

        # backpropogate
        if (game_id + 1) % config['batch_size'] == 0:
            for player_num, player_id in enumerate([player1_id, player2_id]):
                loss[player_num].backward()
                optimizers[player_id].step()

                optimizers[player_id].zero_grad()
                loss[player_num] = torch.tensor(0.0, device=config['device'])
            player1_id, player2_id = torch.randperm(len(players))[:2]

            # pbt update
            if (game_id + 1) % config['pbt_frequency'] == 0:
                players, player_configs, optimizers = pbt_step(
                    config = config,
                    players = players,
                    optimizers = optimizers,
                    player_configs= player_configs,
                    rewards = log['rewards']
                )
                # for player_id in range(len(players)):
                #     optimizers[player_id].update_config(player_configs[player_id])

    return log

if __name__ == '__main__':
    try:
        config = {
            'device': 'cpu',
            'replay_buffer_capacity': 1_000,
            'minibatch_size': 10,
            # 'warmup_steps': 100,
            'train_steps': 10_000,
            'log_window': 1_000,
            'batch_size': 32,
            'pbt_frequency': 1024,
            'sample_size': 100,
            'confidence_level': 0.99,
            "hyperparameter_raw_init_distributions": {
                "epsilon": torch.distributions.Uniform(
                    torch.tensor(-10, device="mps", dtype=torch.float32),
                    torch.tensor(-5, device="mps", dtype=torch.float32)
                ),
                "first_moment_decay": torch.distributions.Uniform(
                    torch.tensor(-3, device="mps", dtype=torch.float32),
                    torch.tensor(0, device="mps", dtype=torch.float32)
                ),
                "learning_rate": torch.distributions.Uniform(
                    torch.tensor(-5, device="mps", dtype=torch.float32),
                    torch.tensor(-1, device="mps", dtype=torch.float32)
                ),
                "second_moment_decay": torch.distributions.Uniform(
                    torch.tensor(-5, device="mps", dtype=torch.float32),
                    torch.tensor(-1, device="mps", dtype=torch.float32)
                ),
                "weight_decay": torch.distributions.Uniform(
                    torch.tensor(-5, device="mps", dtype=torch.float32),
                    torch.tensor(-1, device="mps", dtype=torch.float32)
                )
            },
            "hyperparameter_raw_perturb": {
                "epsilon": torch.distributions.Normal(
                    torch.tensor(0, device="mps", dtype=torch.float32),
                    torch.tensor(1, device="mps", dtype=torch.float32)
                ),
                "first_moment_decay": torch.distributions.Normal(
                    torch.tensor(0, device="mps", dtype=torch.float32),
                    torch.tensor(1, device="mps", dtype=torch.float32)
                ),
                "learning_rate": torch.distributions.Normal(
                    torch.tensor(0, device="mps", dtype=torch.float32),
                    torch.tensor(1, device="mps", dtype=torch.float32)
                ),
                "second_moment_decay": torch.distributions.Normal(
                    torch.tensor(0, device="mps", dtype=torch.float32),
                    torch.tensor(1, device="mps", dtype=torch.float32)
                ),
                "weight_decay": torch.distributions.Normal(
                    torch.tensor(0, device="mps", dtype=torch.float32),
                    torch.tensor(1, device="mps", dtype=torch.float32)
                ),
            },
            "hyperparameter_transforms": {
                "epsilon": lambda log10: 10 ** log10,
                "first_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
                "learning_rate": lambda log10: 10 ** log10,
                "second_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
                "weight_decay": lambda log10: 10 ** log10,
            },
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
        player3 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player4 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player5 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player6 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player7 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )
        player8 = Sequential(
            Linear(9, 12),
            ReLU(),
            Linear(12, 10),
            ReLU(),
            Linear(10, 6),
            ReLU(),
            Linear(6,2)
        )

        players = [
            player1,
            player2,
            player3,
            player4,
            player5,
            player6,
            player7,
            player8
        ]
        player_configs = [
            dict(config) for _ in range(len(players))
        ]

        # train
        log = train(config, players, player_configs)
        # pdb.set_trace()

        fig, ax = plt.subplots(2,2)
        colors = ['red','orange','yellow', 'green', 'blue', 'purple','teal', 'violet']

        for player_id in range(len(players)):
            # print analysis
            print(f'analysis for player {player_id}')
            analyze_strategy(players[player_id])

            # plot average rewards
            kernel = torch.ones(config['log_window']) / config['log_window']
            rewards = torch.tensor(log['rewards'][player_id], dtype=torch.float32)
            average_score = torch.conv1d(
                rewards.view(1,1,-1),
                kernel.view(1,1,-1)
            ).view(-1)
            ax[0,0].plot(average_score, label = f'{player_id} rewards', c = colors[player_id])
            # ax[0,0].axhline(y=float(1/18), color='r', linestyle='--', label='Nash (P1 = 1/18)')
            
            # ax[0,0].set_yscale('log')

            ax[0,1].plot(torch.cumsum(rewards, dim = 0), label = f'{player_id} score',c = colors[player_id])
        fig.legend()
        plt.show()

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
