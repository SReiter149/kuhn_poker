
import os
import sys
import yaml
import json
import numpy as np
import torch
from datetime import datetime
import argparse
import pdb
from pathlib import Path

from matplotlib import pyplot as plt

from train import train, train_vs_optimal_bot, train_self_play
from game import KuhnPoker
from models import create_kuhn_player
from analyze import analyze_strategy, distance_from_optimal


from plot_research import (
    plot_training_convergence,
    plot_strategy_heatmap,
    plot_strategy_heatmap_p2,
    plot_strategy_comparison_nash,
    plot_players_comparison,
    plot_multiple_dstance_learning
)

def create_output_directory(experiment_number = 1, base_dir="./results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"experiment{experiment_number}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir)

def save_results(output_dir, players, log, config, metrics):
        
    
    for i, player in enumerate(players):
        model_path = os.path.join(output_dir, f"player_{i}_model.pt")
        torch.save(player.state_dict(), model_path)
    print(f"✓ Saved {len(players)} player models")
    
    
    rewards_path = os.path.join(output_dir, "rewards_log.npz")
    np.savez(rewards_path, 
             rewards=log['rewards'].cpu().numpy(),
             player_ids=log['player_ids'].cpu().numpy())
    print(f"✓ Saved rewards log: {rewards_path}")
    
    
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Saved config: {config_path}")
    
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")

def print_summary(metrics):
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Training Episodes:        {metrics['total_episodes']:,}")
    print(f"Number of Players:        {metrics['num_players']}")
    
    if metrics['num_players'] == 1:
        print(f"\nPlayer 0 Performance:")
        print(f"  Final Avg Reward:       {metrics['player_0']['final_avg_reward']:.4f}")
        print(f"  Nash Equilibrium:       -0.0556")
        print(f"  Distance from Nash:     {abs(metrics['player_0']['final_avg_reward'] + 0.0556):.4f}")
    else:
        print("\nPlayers Performance:")
        for i in range(metrics['num_players']):
            player_metrics = metrics[f'player_{i}']
            print(f"  Player {i}:")
            print(f"    Final Avg Reward:     {player_metrics['final_avg_reward']:.4f}")
            print(f"    Total Games:          {player_metrics['num_games']}")
    
    print("="*70)

def analyze_single_player_strategy(player, device='cpu'):
    player.eval()
   
    def get_bet_prob(state_idx):
        state = torch.zeros(12, dtype=torch.float32, device=device)
        state[state_idx] = 1
        
        with torch.no_grad():
            prob = player(state)
            
            if isinstance(prob, torch.Tensor):
                return prob.item()
            return prob
    
    metrics = {
        'jack_bet': get_bet_prob(0),      
        'queen_bet': get_bet_prob(1),     
        'king_bet': get_bet_prob(2),      
        'jack_call': get_bet_prob(6),     
        'queen_call': get_bet_prob(7),    
        'king_call': get_bet_prob(8),     
    }
    
    
    metrics['alpha'] = metrics['jack_bet']
    
    return metrics

def run_experiment3(config_path = "config3.yaml"):
    # set up train
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config['seed'])

    train_config = config['training']
    game = KuhnPoker()

    num_players = train_config['num_players']
    players = [create_kuhn_player(train_config['device']) for i in range(num_players)]
    logs = []

        # train
    for player_id in range(num_players):
        player, train_log = train_self_play(train_config, players[player_id], game)
        logs.append(train_log)
        players[player_id] = player
        print(distance_from_optimal(player))

    # analysis
    output_dir = create_output_directory(3)

    final_distances = np.array([logs[player_id]['distances'][-1] for player_id in range(num_players)])
    best_model_id = np.argmin(final_distances)

    

    # plot learning of all
    plot_multiple_dstance_learning(logs, output_dir / "overview.png")
    plot_strategy_heatmap(players[best_model_id], save_path= output_dir / "p1_strategy_heatmap.png")
    plot_strategy_heatmap_p2(players[best_model_id], save_path= output_dir / "p2_strategy_heatmap.png")
    plot_training_convergence((-1)*logs[best_model_id]['reward'], save_path= output_dir / "training_convergence.png")

    pdb.set_trace()

def run_experiment2(config_path = "config2.yaml"):   
    # set up train
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config['seed'])

    train_config = config['training']
    game = KuhnPoker()

    num_players = train_config['num_players']
    players = [create_kuhn_player(train_config['device']) for i in range(num_players)]
    logs = []

    # train
    for player_id in range(num_players):
        player, train_log = train_vs_optimal_bot(train_config, players[player_id], game)
        logs.append(train_log)
        players[player_id] = player
        print(distance_from_optimal(player))

    # analysis
    output_dir = create_output_directory(2)

    final_distances = np.array([logs[player_id]['distances'][-1] for player_id in range(num_players)])
    best_model_id = np.argmin(final_distances)

    pdb.set_trace()

    # plot learning of all
    plot_multiple_dstance_learning(logs, output_dir / "overview.png")
    plot_strategy_heatmap(players[best_model_id], save_path= output_dir / "strategy_heatmap.png")
    plot_training_convergence((-1)*logs[best_model_id]['reward'], save_path= output_dir / "training_convergence.png")

def run_experiment(config_path="config.yaml", output_dir=None, quick_mode=False, num_players=None):
        
    print("="*70)
    print("KUHN POKER RL EXPERIMENT (Multi-Agent Self-Play)")
    print("="*70)
    
    
    print("\n[1/7] Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    
    if 'training' in config:
        
        train_config = {
            'device': config['training']['device'],
            'train_steps': config['training']['train_steps'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['lr'],
            'log_window': config['training']['log_window']
        }
    else:
        
        train_config = config.copy()
    
    
    if num_players is not None:
        train_config['num_players'] = num_players
    elif 'num_players' not in train_config:
        train_config['num_players'] = 2  
    
    
    if quick_mode:
        original_steps = train_config['train_steps']
        train_config['train_steps'] = min(10000, original_steps)
        print(f"  QUICK MODE: Reduced training from {original_steps:,} to {train_config['train_steps']:,} steps")
    
    print(f"  Device: {train_config['device']}")
    print(f"  Training steps: {train_config['train_steps']:,}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Number of players: {train_config['num_players']}")
    
    
    print("\n[2/7] Initializing game and players...")
    game = KuhnPoker()
    
    
    num_players = train_config['num_players']
    players = [create_kuhn_player(train_config['device']) for i in range(num_players)]
    print(f"  ✓ Created {num_players} players")
    
    
    print("\n[3/7] Training models...")
    
    if num_players == 1:
        
        print("  Mode: Training P1 vs Optimal Nash Equilibrium Bot (P2)")
        log = train_vs_optimal_bot(train_config, players[0], game)
    else:
        
        print("  Mode: Multi-Agent Self-Play (Round-Robin)")
        print("  (This may take a few minutes...)")
        
        
        import train as train_module
        train_module.kuhn = game
        train_module.config = train_config
        train_module.num_players = num_players
        
        log = train(train_config, players)
    
    print(f"  ✓ Training complete! ({train_config['train_steps']:,} episodes)")
    
    
    print("\n[4/7] Analyzing learned strategies...")
    
    
    player_metrics = {}
    for player_id in range(num_players):
        mask = (log['player_ids'] == player_id).cpu().numpy()
        player_rewards = log['rewards'][mask].cpu().numpy()
        
        if len(player_rewards) > 0:
            final_avg = np.mean(player_rewards[-min(100, len(player_rewards)):])
            
            
            strategy_metrics = analyze_single_player_strategy(players[player_id], train_config['device'])
            
            player_metrics[f'player_{player_id}'] = {
                'final_avg_reward': float(final_avg),
                'num_games': int(len(player_rewards)),
                'strategy': strategy_metrics
            }
            
            print(f"  Player {player_id}:")
            print(f"    Final avg reward: {final_avg:.4f}")
            print(f"    Alpha (bluff freq): {strategy_metrics['alpha']:.4f}")
    
    
    metrics = {
        'total_episodes': int(train_config['train_steps']),
        'num_players': num_players,
        'config': train_config,
        'timestamp': datetime.now().isoformat()
    }
    metrics.update(player_metrics)
    
    print_summary(metrics)
    
    
    print(f"\n[5/7] Creating output directory...")
    if output_dir is None:
        output_dir = create_output_directory()
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"  {output_dir}")
    
    
    print("\n[6/7] Saving results...")
    save_results(output_dir, players, log, train_config, metrics)
    
    
    print("\n[7/7] Generating plots...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if num_players == 1:
        
        player_id = 0
        mask = (log['player_ids'] == player_id).cpu().numpy()
        rewards = log['rewards'][mask].cpu().numpy().tolist()
        
        print("  [1/4] Training convergence...")
        plot_training_convergence(
            rewards,
            window=train_config.get('log_window', 1000),
            save_path=os.path.join(plots_dir, "01_training_convergence.png")
        )
        
        print("  [2/4] Strategy heatmap...")
        plot_strategy_heatmap(
            players[0],
            device=train_config['device'],
            save_path=os.path.join(plots_dir, "02_strategy_heatmap.png")
        )
        
        print("  [3/4] Nash equilibrium comparison...")
        plot_strategy_comparison_nash(
            players[0],
            device=train_config['device'],
            save_path=os.path.join(plots_dir, "03_nash_comparison.png")
        )
        
        print("  [4/4] Comprehensive analysis...")
        
        plot_config = {
            'training': {
                'device': train_config['device'],
                'log_window': train_config.get('log_window', 1000)
            },
            'game': {'p2_deterministic': False}
        }
    else:
        
        print("  [1/2] Players comparison...")
        plot_players_comparison(
            player_configs=[{} for _ in players],
            rewards_log=log,
            save_path=os.path.join(plots_dir, "01_players_comparison.png")
        )
        
        print("  [2/2] Individual player strategies...")
        for i, player in enumerate(players):
            try:
                plot_strategy_heatmap(
                    player,
                    device=train_config['device'],
                    save_path=os.path.join(plots_dir, f"player_{i}_strategy.png")
                )
            except Exception as e:
                print(f"    Warning: Could not plot player {i} strategy: {e}")
    
    
    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write("KUHN POKER RL EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        f.write(f"Training Episodes: {metrics['total_episodes']:,}\n")
        f.write(f"Number of Players: {metrics['num_players']}\n\n")
        
        for player_id in range(num_players):
            player_key = f'player_{player_id}'
            if player_key in metrics:
                pm = metrics[player_key]
                f.write(f"Player {player_id}:\n")
                f.write(f"  Final Avg Reward: {pm['final_avg_reward']:.4f}\n")
                f.write(f"  Games Played: {pm['num_games']}\n")
                if 'strategy' in pm:
                    f.write(f"  Strategy Metrics:\n")
                    for key, val in pm['strategy'].items():
                        f.write(f"    {key}: {val:.4f}\n")
                f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("\nFiles Generated:\n")
        for i in range(num_players):
            f.write(f"  - player_{i}_model.pt\n")
        f.write("  - rewards_log.npz\n")
        f.write("  - config.yaml\n")
        f.write("  - metrics.json\n")
        if num_players == 1:
            f.write("  - plots/01_training_convergence.png\n")
            f.write("  - plots/02_strategy_heatmap.png\n")
            f.write("  - plots/03_nash_comparison.png\n")
            f.write("  - plots/04_comprehensive_analysis.png\n")
        else:
            f.write("  - plots/01_players_comparison.png\n")
            for i in range(num_players):
                f.write(f"  - plots/player_{i}_strategy.png\n")
    
    print(f"  ✓ Saved summary: {summary_path}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("="*70 + "\n")
    
    return output_dir, metrics

def main():
    parser = argparse.ArgumentParser(
        description='Run Kuhn Poker RL experiment using train.py and game.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: ./results/experiment_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduce training steps to 10k for testing'
    )
    
    parser.add_argument(
        '--num-players',
        type=int,
        default=None,
        help='Number of players in multi-agent self-play (default: 2)'
    )
    
    args = parser.parse_args()
    
    try:
        output_dir, metrics = run_experiment(
            config_path=args.config,
            output_dir=args.output_dir,
            quick_mode=args.quick,
            num_players=args.num_players
        )
        
        return 0
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        run_experiment3()
    except Exception as e:
        import traceback
        traceback.print_exc()
        pdb.post_mortem()
    # sys.exit(main())
