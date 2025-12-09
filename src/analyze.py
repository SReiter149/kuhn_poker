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
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in range(len(players))]

    for player_id in range(len(players)):        
        # visualize
        as_player1 = log['player1_ids'] == player_id
        as_player2 = log['player2_ids'] == player_id

        # model’s reward at each timestep, with sign
        signed_all = torch.where(as_player1, 
                                log['rewards'],        # when P1: +reward
                                -log['rewards'])       # otherwise: -reward

        # but keep only timesteps where it actually played (P1 or P2)
        mask = as_player1 | as_player2
        rewards = signed_all[mask]

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