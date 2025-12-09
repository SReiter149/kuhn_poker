import torch
from matplotlib import pyplot as plt

def analyze_strategy(player):
    for i in range(12):
        state = [0] * 12
        state[i] = 1
        state = torch.tensor(state, dtype = torch.float32)

        prob = player(state)
        print(f'given state {state.detach().cpu().numpy()} prob of call/bet is {prob.detach().item()}')


def plot_training(config, players,log):
    fig, ax = plt.subplots(2,2)
    cmap = plt.get_cmap('tab20') # viridis
    colors = [cmap(i) for i in range(len(players))]

    for player_id in range(len(players)):        
        # visualize
        mask = log['player_ids'] == player_id

        rewards = log['rewards'][mask]

        # plot average rewards
        kernel = torch.ones(config['log_window']) / config['log_window']
        rewards = torch.tensor(rewards, dtype=torch.float32)
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