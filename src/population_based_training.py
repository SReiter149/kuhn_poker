import torch
import copy 

from util import welch_one_sided
from util import get_seed, get_mlp, AdamW

from collections.abc import (
    Callable,
)

import pdb

def pbt_init(
    player_configs
):
    for player_id in range(len(player_configs)):
        for name, distribution in player_configs[player_id]['hyperparameter_raw_init_distributions'].items():
            raw = distribution.sample()
            player_configs[player_id][name + '_raw'] = raw
            player_configs[player_id][name] = player_configs[player_id]['hyperparameter_transforms'][name](raw)
    
    return player_configs

def pbt_step(
    config,
    players,
    optimizers,
    player_configs,
    rewards,
):
    device = config['device']
    player_count = len(players)
    sample_size = config['sample_size']
    for i in range(len(rewards)):
        sample_size = min(sample_size, len(rewards[i]))
    assert sample_size > 0
    confidence_level = config['confidence_level']
    eval_rewards = torch.stack([torch.tensor(rewards[i][-sample_size:]) for i in range(player_count)]).to(device = device).transpose(0,1)
    replacement_indices = torch.randperm(
        player_count,
        device = device 
    )
    mask = welch_one_sided(
        eval_rewards,
        eval_rewards[replacement_indices],
        confidence_level = confidence_level
    )
    if mask.any() == True:
        print('we have an update!')
        for player_id in mask.nonzero(as_tuple=True)[0]:
            players[player_id] = copy.deepcopy(players[replacement_indices[player_id]])
            for name, transform in config['hyperparameter_transforms'].items():
                raw = player_configs[player_id][name + '_raw']
                noise = config['hyperparameter_raw_perturb'][name].sample()
                values_raw = raw + noise
                values = transform(values_raw)
                player_configs[player_id][name] = values
            optimizers[player_id] = AdamW(
                players[player_id].parameters(),
                player_configs[player_id],
            )
    return players, player_configs, optimizers