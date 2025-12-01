from torch.nn import Linear, Sequential
import torch

# from model import Player1, Player2
from game import KuhnPoker

import pdb
from rich.console import Console

if __name__ == '__main__':
    try:
        device = 'cpu'
        player1 = Sequential(
            Linear(12, 10),
            Linear(10, 10),
            Linear(10, 2)
        )

        player2 = Sequential(
            Linear(12, 10),
            Linear(10, 10),
            Linear(10, 2)
        )

        kuhn = KuhnPoker()
        state,terminal = kuhn.reset()
        state = torch.asarray(state, device = device, dtype = torch.float32)
        turn = 0
        terminal = False
        while not terminal:
            if turn % 2 == 0:
                action_distribution = player1(state)
                state, reward, terminal = kuhn.step(torch.argmax(action_distribution))
                state = torch.asarray(state, device = device, dtype = torch.float32)
            else:
                action_distribution = player2(state)
                state, reward, terminal = kuhn.step(torch.argmax(action_distribution))
                state = torch.asarray(state, device = device, dtype = torch.float32)
            turn += 1
        
        print(state)
        print(reward)
        print(terminal)

        # loss1, loss2 = kuhn.play(player1, player2)
        # loss1.backward()
        # loss2.backward()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
