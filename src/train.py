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

class ReplayBuffer:
    def __init__(
        self,
        config: dict,
        observation_shape: torch.Tensor,
    ):
        """
        A replay buffer for DQN.

        Parameters
        ----------
        config : `dict`
            Configuration dictionary. Required key-value pairs:
            `"replay_buffer_capacity"` : `int`
                Capacity of the replay buffer.
            `"device"` : `torch.device | int | str`
                Device to store the replay buffer tensors on.
            `"minibatch_size"` : `int`
                Minibatch size for sampling from the replay buffer.
        observation_shape : 'torch.Int64'
            the shape of the observation shape
        """
        # config stuff
        self.buffer_capacity = config['replay_buffer_capacity']
        self.device = config['device']
        self.minibatch_size = config['minibatch_size']
        self.config = config

        # setup stuff

        self.cursor = 1
        self.size = 0

        # set up storage tensors
        self.actions = torch.empty((self.buffer_capacity,), device = self.device, dtype = torch.int64)
        self.observations = torch.empty((self.buffer_capacity,) + observation_shape, device = self.device, dtype = torch.float32)
        self.rewards = torch.empty((self.buffer_capacity,), device = self.device, dtype = torch.float32)
        self.terminal = torch.empty((self.buffer_capacity,), device = self.device, dtype = torch.bool)


    def get_minibatch(self) -> dict:
        """
        Sample a minibatch from the replay buffer.

        Returns
        -------
        A dictionary with the following key-value pairs:
        - `"actions"` : `torch.Tensor`
            A tensor of shape `(minibatch_size,)`
            of actions.
        - `"observations"` : `torch.Tensor`
            A tensor of shape `(minibatch_size, observation_dim)`
            of observations. These are the observations preceding the actions.
        - `"observations_next"` : `torch.Tensor`
            A tensor of shape `(minibatch_size, observation_dim)`
            of next observations. These are the observations following the actions.
        - `"rewards"` : `torch.Tensor`
            A tensor of shape `(minibatch_size,)`
            of rewards.
        - `"terminal"` : `torch.Tensor`
            A tensor of shape `(minibatch_size,)`
            of terminal flags.
        """
        indices = torch.randint(0, self.size, (self.minibatch_size,), device = self.device)
        indices += self.cursor - self.size
        indices %= self.buffer_capacity

        actions = torch.gather(self.actions, dim = 0, index = indices)
        observations = torch.gather(
            self.observations,
            dim=0,
            index = indices.unsqueeze(-1)
            )

        observations_next = torch.gather(
            self.observations,
            dim=0,
            index=indices.unsqueeze(-1).expand(
                *indices.shape,
                self.observations.shape[-1]
            )
        )
        rewards = torch.gather(
            self.rewards,
            dim=0,
            index=indices
        )
        terminal = torch.gather(
            self.terminal,
            dim=0,
            index=indices
        )

        minibatch = {
            "actions": actions,
            "observations": observations,
            "observations_next": observations_next,
            "rewards": rewards,
            "terminal": terminal,
        }

        return minibatch


    def update(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor
    ):
        """
        Store new step data at the cursor, then update the cursor and the size. Note that the maximum value of the size is capacity minus 1.

        Parameters
        ----------
        actions : `torch.int64`
            A tensor of shape `` of actions.
        observations : `torch.Tensor`
            A tensor of shape ` + (observation_dim,)` of observations.
        rewards : `torch.float32`
        terminal : `torch.bool`
        """
        assert action.dtype == torch.int64
        assert reward.dtype == torch.float32
        assert terminal.dtype == torch.bool

        self.actions[self.cursor] = action
        self.observations[self.cursor] = observation
        self.rewards[self.cursor] = reward
        self.terminal[self.cursor] = terminal

        self.cursor += 1
        self.cursor %= self.buffer_capacity

        self.size = max(self.buffer_capacity - 1, self.size + 1)

def play_game(players, game):
        
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
        action_probabilities = players[player](state)

        dist = Categorical(logits=action_probabilities)
        action = dist.sample()                         # shape (1,)
        log_prob = dist.log_prob(action).squeeze(0)    # scalar
        # pdb.set_trace()
        state, reward, terminal = game.step(action.numpy())

        state = torch.asarray(state, device = config['device'], dtype = torch.float32)
        reward = torch.asarray(reward, device = config['device'], dtype = torch.float32)
        terminal = torch.asarray(terminal, device = config['device'], dtype = torch.bool)

        # update last players buffer based on current players move
        if turn >= 1:
            assert len(last_log_prob.shape) == 0 
            memory[(player + 1) % 2]['probabilities'].append(last_log_prob)
            memory[(player + 1) % 2]['observations'].append(last_state)
            memory[(player + 1) % 2]['rewards'].append(reward[(player + 1) % 2])

        last_log_prob = log_prob
        last_state = state
        turn += 1

    assert len(last_log_prob.shape) == 0 
    memory[player]['probabilities'].append(last_log_prob)
    memory[player]['observations'].append(last_state)
    memory[player]['rewards'].append(reward[player])
    
    for player in range(2):
        for key in memory[player].keys():
            # pdb.set_trace()
            memory[player][key] = torch.stack(memory[player][key])
    return memory


def train(config, players):
    optimizers = [AdamW(players[i].parameters()) for i in range(len(players))]
    log = {
        'player1_rewards': [],
        'player1_score': []
    }

    progress_bar = tqdm(range(config['train_steps']))
    for game_id in progress_bar:
        memory = play_game(players, kuhn)
        log['player1_rewards'].append(memory[0]['rewards'][-1])
        if game_id >= 1:
            log['player1_score'].append(log['player1_score'][-1] + log['player1_rewards'][-1])
        else:
            log['player1_score'].append(log['player1_rewards'][-1])
            

        for player in range(len(players)):
            if len(memory[player]['rewards']) > 1:
                rewards = torch.tensor([sum(memory[player]['rewards'][move:]) for move in range(len(memory[player]['rewards']))], device= config['device'], dtype = torch.float32)
            else:
                rewards = memory[player]['rewards'][0]
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

        log = train(config, players)
        
        plt.plot(log['player1_score'])
        plt.show()

        pdb.set_trace()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
