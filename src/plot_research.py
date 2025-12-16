import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import yaml

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
ALPHA_CHAR = '\u03B1'

def plot_multiple_dstance_learning(logs, save_path):
    final_distances = np.array([logs[player_id]['distances'][-1] for player_id in range(len(logs))])

    # plot learning of all
    for p in np.linspace(0, 1, 11):
        val = np.quantile(final_distances, p)
        idx = np.argmin(np.abs(final_distances - val))
        plt.plot(logs[idx]['distances'], label = f"{p:.2f}")
        # pdb.set_trace()
    plt.legend(title = "percentile")
    plt.title("training steps VS distance")
    plt.xlabel("number of time back prop happened")
    plt.ylabel("the distance from the optimal solution")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def exponential_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    ema = np.zeros_like(data)
    alpha = 2 / (window + 1)
    ema[0] = data[0]
    
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    
    return ema


def plot_training_convergence(
    rewards: List[float],
    window: int = 1000,
    nash_eq: float = -1/18,
    title: str = "Training Convergence to Nash Equilibrium",
    save_path: Optional[str] = None
):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rewards_arr = np.array(rewards)
    episodes = np.arange(len(rewards_arr))

    smoothed = exponential_moving_average(rewards_arr, window)

    rolling_std = np.array([
        np.std(rewards_arr[max(0, i-window):i+1]) 
        for i in range(len(rewards_arr))
    ])
    
    ax.plot(episodes, rewards_arr, alpha=0.15, color='gray', 
            linewidth=0.5, label='Raw rewards')
    
    ax.plot(episodes, smoothed, linewidth=2.5, color='#2E86AB', 
            label=f'EMA (window={window})')
    
    ax.fill_between(episodes, 
                     smoothed - rolling_std, 
                     smoothed + rolling_std,
                     alpha=0.2, color='#2E86AB', 
                     label='±1 std')
    
    ax.axhline(y=nash_eq, color='#A23B72', linestyle='--', 
               linewidth=2, label=f'Nash Eq (P1 = {nash_eq:.4f})')
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('P1 Reward', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.set_yscale("symlog", linthresh=1e-3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_strategy_heatmap_p2(
    player_model,
    device: str = 'cpu',
    save_path = None
):

    def get_action_probs(card_idx: int, offset: int):
        state = np.zeros(12, dtype=np.float32)
        state[offset + card_idx] = 1
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        prob_one = player_model(state_t)
        prob_zero = 1.0 - prob_one
        return np.array([prob_zero, prob_one])

    cards = ['Jack', 'Queen', 'King']
    panels = [
        ('Response to P1 Bet', 3),
        ('After P1 Pass (P2 decision)', 9),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (title, offset) in enumerate(panels):
        ax = axes[idx]
        probs_matrix = np.zeros((3, 2))
        # pdb.set_trace()
        for card_idx, _ in enumerate(cards):
            probs_matrix[card_idx] = get_action_probs(card_idx, offset).squeeze(1)

        im = ax.imshow(probs_matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Fold/Pass', 'Call/Bet'], fontsize=11)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(cards, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

        for i in range(3):
            for j in range(2):
                ax.text(j, i, f'{probs_matrix[i, j]:.3f}',
                        ha='center', va='center',
                        color='black' if probs_matrix[i, j] > 0.5 else 'white',
                        fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=11, fontweight='bold')

    plt.suptitle('P2 Strategy Heatmap', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

def plot_strategy_heatmap(
    player_model,
    device: str = 'cpu',
    save_path: Optional[str] = None
):
    player_model.eval()
    
    def get_action_probs(card_idx, is_response=False):
        state = np.zeros(12, dtype=np.float32)
        if not is_response:
            state[card_idx] = 1
        else:
            state[6 + card_idx] = 1
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob_bet = player_model(state_t).item()
            prob_pass = 1.0 - prob_bet
        
        return np.array([prob_pass, prob_bet])
    
    cards = ['Jack', 'Queen', 'King']
    initial_states = [
        ('Initial Bet', False),
        ('Response to P2 Bet', True)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (state_name, history) in enumerate(initial_states):
        ax = axes[idx]
        
        probs_matrix = np.zeros((3, 2))
        for card_idx, card_name in enumerate(cards):
            probs = get_action_probs(card_idx, history)
            probs_matrix[card_idx] = probs
        
        im = ax.imshow(probs_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=0, vmax=1, interpolation='nearest')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Fold/Pass', 'Call/Bet'], fontsize=11)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(cards, fontsize=11)
        ax.set_title(state_name, fontsize=13, fontweight='bold', pad=10)
        
        for i in range(3):
            for j in range(2):
                text = ax.text(j, i, f'{probs_matrix[i, j]:.3f}',
                             ha="center", va="center", 
                             color="black" if probs_matrix[i, j] > 0.5 else "white",
                             fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=11, fontweight='bold')
    
    plt.suptitle('P1 Strategy Heatmap', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_exploitability_evolution(
    exploitability_values: List[float],
    checkpoints: List[int],
    save_path: Optional[str] = None
):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    nash_ev = -1/18
    
    ax.plot(checkpoints, exploitability_values, 
            marker='o', markersize=8, linewidth=2.5, 
            color='#F18F01', label='Measured EV')
    
    ax.axhline(y=nash_ev, color='#A23B72', linestyle='--', 
               linewidth=2, label=f'Nash EV ({nash_ev:.4f})')
    
    y_min, y_max = ax.get_ylim()
    ax.fill_between(checkpoints, nash_ev, y_max, 
                     alpha=0.15, color='red', label='Exploitable Region')
    ax.fill_between(checkpoints, y_min, nash_ev, 
                     alpha=0.15, color='green', label='Sub-Nash Region')
    
    ax.set_xlabel('Training Episodes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Expected Value (P1)', fontsize=13, fontweight='bold')
    ax.set_title('Exploitability Evolution During Training', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_players_comparison(
    player_configs: List[Dict],
    rewards_log: Dict,
    save_path: Optional[str] = None
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    num_players = len(player_configs)
    colors = sns.color_palette("husl", num_players)
    
    player_ids = rewards_log['player_ids'].cpu().numpy()
    rewards = rewards_log['rewards'].cpu().numpy()
    
    ax = axes[0, 0]
    window = 500
    for player_id in range(num_players):
        mask = player_ids == player_id
        player_rewards = rewards[mask]
        if len(player_rewards) > 0:
            smoothed = exponential_moving_average(player_rewards, window)
            ax.plot(smoothed, linewidth=2, color=colors[player_id], 
                   label=f'Player {player_id}', alpha=0.8)
    
    ax.axhline(y=-1/18, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Nash Eq')
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reward (EMA)', fontsize=11, fontweight='bold')
    ax.set_title('Multi-Agent Reward Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for player_id in range(num_players):
        mask = player_ids == player_id
        player_rewards = rewards[mask]
        if len(player_rewards) > 0:
            cumulative = np.cumsum(player_rewards)
            ax.plot(cumulative, linewidth=2, color=colors[player_id], 
                   label=f'Player {player_id}', alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Score', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Performance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    reward_distributions = []
    labels = []
    for player_id in range(num_players):
        mask = player_ids == player_id
        player_rewards = rewards[mask]
        if len(player_rewards) > 0:
            reward_distributions.append(player_rewards)
            labels.append(f'P{player_id}')
    
    positions = range(len(reward_distributions))
    bp = ax.boxplot(reward_distributions, positions=positions, 
                    patch_artist=True, labels=labels,
                    widths=0.6, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=-1/18, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Nash Eq')
    ax.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax.set_title('Reward Distribution by Player', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    
    ax = axes[1, 1]
    win_counts = []
    total_counts = []
    for player_id in range(num_players):
        mask = player_ids == player_id
        player_rewards = rewards[mask]
        if len(player_rewards) > 0:
            win_counts.append(np.sum(player_rewards > 0))
            total_counts.append(len(player_rewards))
    
    win_rates = [w/t if t > 0 else 0 for w, t in zip(win_counts, total_counts)]
    bars = ax.bar(range(num_players), win_rates, color=colors, alpha=0.8)
    
    ax.set_xlabel('Player ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax.set_title('Player Win Rates', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticks(range(num_players))
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Multi-Agent Self-Play Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_strategy_comparison_nash(
    player_model,
    device: str = 'cpu',
    save_path: Optional[str] = None
):
    player_model.eval()
    
    def get_bet_prob(card_idx, state_offset):
        state = np.zeros(12, dtype=np.float32)
        state[state_offset + card_idx] = 1
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob_bet = player_model(state_t).item()
        
        return prob_bet
    
    # Initial betting (states 0-2): Jack=0, Queen=1, King=2
    learned_bet_j = get_bet_prob(0, 0)  # State 0
    learned_bet_q = get_bet_prob(1, 0)  # State 1
    learned_bet_k = get_bet_prob(2, 0)  # State 2
    
    # Response to P2 bet after P1 pass (states 6-8): Jack=6, Queen=7, King=8
    learned_call_j = get_bet_prob(0, 6)  # State 6
    learned_call_q = get_bet_prob(1, 6)  # State 7
    learned_call_k = get_bet_prob(2, 6)  # State 8
    
    alpha_learned = learned_bet_j
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Initial betting probabilities
    ax = axes[0]
    cards = ['Jack', 'Queen', 'King']
    bet_probs = [learned_bet_j, learned_bet_q, learned_bet_k]
    
    bars = ax.bar(cards, bet_probs, color=['#E63946', '#F4A261', '#2A9D8F'], alpha=0.8)
    
    ax.set_ylabel('Bet Probability', fontsize=12, fontweight='bold')
    ax.set_title('Initial Betting Strategy', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, bet_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right panel: Calling probabilities after P2 bet
    ax = axes[1]
    call_probs = [learned_call_j, learned_call_q, learned_call_k]
    
    bars = ax.bar(cards, call_probs, color=['#E63946', '#F4A261', '#2A9D8F'], alpha=0.8)
    
    ax.set_ylabel('Call Probability', fontsize=12, fontweight='bold')
    ax.set_title('Response to P2 Bet (after P1 Pass)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, call_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add alpha info
    in_nash_range = 0 <= alpha_learned <= 1/3
    status = "Within Nash Range [0, 1/3]" if in_nash_range else "Outside Nash Range"
    fig.text(0.5, 0.02, f'Bluffing Parameter α = {alpha_learned:.4f} ({status})', 
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', 
                     facecolor='lightgreen' if in_nash_range else 'wheat', 
                     alpha=0.5))
    
    plt.suptitle('Learned Strategy', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()