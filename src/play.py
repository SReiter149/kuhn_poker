import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

char_alpha = '\u03B1'

class KuhnPoker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.history_log = [-1, -1, -1]  # [P1_Action, P2_Action, P1_Action_2]
        self.cards = np.random.choice(3, size=2, replace=False)  # 0=J, 1=Q, 2=K
        return self.get_state(), False

    def get_state(self):
        player_idx = self.turn % 2
        norm_card = self.cards[player_idx] / 2.0 
        
        # One-hot encode history: -1 -> [1,0,0], 0 -> [0,1,0], 1 -> [0,0,1]
        encoded_history = []
        for action in self.history_log:
            vec = [0, 0, 0]
            if action == -1: vec[0] = 1
            elif action == 0: vec[1] = 1
            elif action == 1: vec[2] = 1
            encoded_history.extend(vec)
            
        return np.array([norm_card] + encoded_history, dtype=np.float32)

    def step(self, action):
        action = int(action)
        self.history_log[self.turn] = action
        
        terminal, winner, pot = self._check_terminal()
        
        if terminal:
            rewards = np.zeros(2)
            if winner == 0:
                if self.cards[0] > self.cards[1]: rewards = [pot, -pot]
                else:                             rewards = [-pot, pot]
            elif winner == 1:
                rewards = [pot, -pot]
            elif winner == 2:
                rewards = [-pot, pot]
            return self.get_state(), rewards, True

        self.turn += 1
        return self.get_state(), [0, 0], False

    def _check_terminal(self):
        h = self.history_log
        if h[0]==0 and h[1]==0: return True, 0, 1
        if h[0]==1 and h[1]==0: return True, 1, 1
        if h[0]==1 and h[1]==1: return True, 0, 2
        if h[0]==0 and h[1]==1 and h[2]==0: return True, 2, 1
        if h[0]==0 and h[1]==1 and h[2]==1: return True, 0, 2
        return False, None, None

class NeuralPlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), 
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): 
        return self.net(x)

class OptimalBotP2:
    """Hardcoded GTO strategy for Player 2 in Kuhn Poker."""
    def __call__(self, state_tensor):
        state = state_tensor.cpu().numpy()
        card_norm = state[0]
        
        def decode_action(start_idx):
            segment = state[start_idx : start_idx + 3]
            idx = np.argmax(segment)
            if idx == 0: return -1
            elif idx == 1: return 0
            elif idx == 2: return 1
            return -1
        
        p1_first_action = decode_action(1)
        p1_second_action = decode_action(4)
        p2_action = decode_action(7)
        
        card = int(round(card_norm * 2))
        logits = torch.zeros(2)
        
        if p1_first_action == 1:
            if card == 0:  # Jack: Always Fold
                logits = torch.tensor([10.0, -10.0])
            elif card == 2:  # King: Always Call
                logits = torch.tensor([-10.0, 10.0])
            elif card == 1:  # Queen: Call 1/3, Fold 2/3
                logits = torch.tensor([np.log(2/3), np.log(1/3)])
                    
        elif p1_first_action == 0:
            if card == 0:  # Jack: Bet 1/3, Check 2/3
                logits = torch.tensor([np.log(2/3), np.log(1/3)])
            elif card == 2:  # King: Always Bet
                logits = torch.tensor([-10.0, 10.0])
            elif card == 1:  # Queen: Always Check
                logits = torch.tensor([10.0, -10.0])
        
        return logits

def train(config):
    env = KuhnPoker()
    device = config['training']['device']
    
    p1 = NeuralPlayer().to(device)
    
    print("--- Mode: Training vs GTO Bot ---")
    p2 = OptimalBotP2() 
    optimizer = AdamW(p1.parameters(), lr=config['training']['lr'])
    
    p1_rewards_history = []
    batch_loss = 0
    batch_entropy = 0
    batch_counter = 0
    BATCH_SIZE = config['training']['batch_size']
    
    optimizer.zero_grad()
    
    pbar = tqdm(range(config['training']['train_steps']))
    for _ in pbar:
        state, _ = env.reset()
        log_probs = []
        entropies = []
        
        terminal = False
        while not terminal:
            p_idx = env.turn % 2
            state_t = torch.FloatTensor(state).to(device)
            
            if p_idx == 0:
                logits = p1(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                act_int = action.item()
            else:
                with torch.no_grad():
                    logits = p2(state_t)
                    dist = Categorical(logits=logits)
                    act_int = dist.sample().item()

            state, rewards, terminal = env.step(act_int)
        
        traj_log_prob = torch.stack(log_probs).sum()
        traj_entropy = torch.stack(entropies).sum()
        
        reward = rewards[0]
        loss = -traj_log_prob * reward
        
        batch_loss += loss
        batch_entropy += traj_entropy
        batch_counter += 1
        p1_rewards_history.append(rewards[0])

        if batch_counter >= BATCH_SIZE:
            total_loss = batch_loss / BATCH_SIZE
                        
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(p1.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            batch_loss = 0
            batch_entropy = 0
            batch_counter = 0

    return p1, p1_rewards_history

def analyze_p1_strategy(player_model, device='cpu'):
    """Probes P1 strategy at key game states and derives alpha parameter."""
    player_model.eval()
    
    def get_bet_prob(card_val, history):
        norm_card = card_val / 2.0
        
        encoded_history = []
        for action in history:
            vec = [0, 0, 0]
            if action == -1: vec[0] = 1
            elif action == 0: vec[1] = 1
            elif action == 1: vec[2] = 1
            encoded_history.extend(vec)
            
        state = np.array([norm_card] + encoded_history, dtype=np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = player_model(state_t)
            probs = torch.softmax(logits, dim=1)
            
        return probs[0][1].item()

    print("\n--- P1 Strategy Analysis ---")

    p_bet_j = get_bet_prob(0, [-1, -1, -1])
    p_bet_q = get_bet_prob(1, [-1, -1, -1])
    p_bet_k = get_bet_prob(2, [-1, -1, -1])
    p_call_j = get_bet_prob(0, [0, 1, -1])
    p_call_q = get_bet_prob(1, [0, 1, -1])
    p_call_k = get_bet_prob(2, [0, 1, -1])

    print(f"P(Bet | Jack): {p_bet_j:.4f}")
    print(f"P(Bet | Queen): {p_bet_q:.4f}")
    print(f"P(Bet | King): {p_bet_k:.4f}")
    print(f"P(Call| Jack): {p_call_j:.4f}")
    print(f"P(Call| Queen): {p_call_q:.4f}")
    print(f"P(Call| King): {p_call_k:.4f}")

    alpha = p_bet_j
    # 
    # print("\n--- Checking Equilibrium Consistency ---")
    print(f"Estimated {char_alpha} (from Jack): {alpha:.4f}")
    
    in_range = 0.0 <= alpha <= (1.0/3.0 + 0.05)
    print(f"Consistency Check 1 (0 <= {char_alpha} <= 1/3): {'PASS' if in_range else 'FAIL'}")
    k_consistency = abs(p_bet_k - (3 * alpha))
    print(f"Consistency Check 2 (K ≈ 3{char_alpha}) | diff  : {k_consistency:.4f}")

    q_call_consistency = abs(p_call_q - (alpha + (1.0/3.0)))
    print(f"Consistency Check 3 (Q ≈ {char_alpha}+1/3) | diff: {q_call_consistency:.4f}")

    return alpha

def calculate_exploitability(p1_model, device='cpu'):
    """
    Calculates P1 EV against perfect best response.
    GTO equilibrium value is -1/18 (-0.0555).
    """
    p1_model.eval()
    
    def get_node_value(history, p1_card, p2_card):
        turn = len([x for x in history if x != -1])
        player = turn % 2
        
        h = history
        if h[0]==0 and h[1]==0: return 1 if p1_card > p2_card else -1
        if h[0]==1 and h[1]==0: return 1
        if h[0]==1 and h[1]==1: return 2 if p1_card > p2_card else -2
        if h[0]==0 and h[1]==1 and h[2]==0: return -1
        if h[0]==0 and h[1]==1 and h[2]==1: return 2 if p1_card > p2_card else -2

        if player == 0:
            norm_card = p1_card / 2.0
            
            encoded_history = []
            for action in history:
                vec = [0, 0, 0]
                if action == -1: vec[0] = 1
                elif action == 0: vec[1] = 1
                elif action == 1: vec[2] = 1
                encoded_history.extend(vec)

            state_vec = np.array([norm_card] + encoded_history, dtype=np.float32)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = p1_model(state_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            h_fold = list(history); h_fold[turn] = 0
            h_bet  = list(history); h_bet[turn] = 1
            
            val_fold = get_node_value(h_fold, p1_card, p2_card)
            val_bet  = get_node_value(h_bet, p1_card, p2_card)
            
            return probs[0] * val_fold + probs[1] * val_bet
            
        else:
            h_fold = list(history); h_fold[turn] = 0
            h_bet  = list(history); h_bet[turn] = 1
            
            val_fold = get_node_value(h_fold, p1_card, p2_card)
            val_bet  = get_node_value(h_bet, p1_card, p2_card)
            
            return min(val_fold, val_bet)

    total_ev = 0
    perms = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    for c1, c2 in perms:
        total_ev += get_node_value([-1, -1, -1], c1, c2)
        
    return total_ev / 6.0

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("=== Training Configuration ===")
    print(yaml.dump(config, default_flow_style=False))

    p1, history = train(config)
    
    def exponential_moving_average(a, window):
        """Compute EMA using alpha = 2 / (window + 1)."""
        a = np.asarray(a, dtype=float)
        ema = np.zeros_like(a)
        alpha = 2 / (window + 1)
        ema[0] = a[0]

        for t in range(1, len(a)):
            ema[t] = alpha * a[t] + (1 - alpha) * ema[t-1]

        return ema

    window = config['training']['log_window']
    smoothed = exponential_moving_average(history, window)

    print("Final P1 Average Reward:", np.mean(history[-100:]))

    alpha = analyze_p1_strategy(p1, config['training']['device'])
    print(f"Estimated Alpha: {alpha:.4f}")
    ev = calculate_exploitability(p1, config['training']['device'])
    print(f"Exploitability EV: {ev:.4f} (Ideal: -0.0556)")
  
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.axhline(y=float(-1/18), color='r', linestyle='--', label='Nash (P1 = -1/18)')
    plt.title(f"Training Results (P2_Fixed={config['game']['p2_deterministic']})")
    plt.xlabel(f"Episodes (Moving Avg Window {window})")
    plt.ylabel("P1 Reward")
    plt.legend()
    plt.show()