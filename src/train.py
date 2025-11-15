from torch.nn import Linear, Sequential

from model import Player1, Player2
from game import KuhnPoker

import pdb
from rich.console import Console

if __name__ == '__main__':
    try:
        player1 = Sequential(
            Linear(2, 10),
            Linear(10, 10),
            Linear(10, 1)
        )

        player2 = Sequential(
            Linear(2, 10),
            Linear(10, 10),
            Linear(10, 1)
        )

        kuhn = KuhnPoker()
        loss1, loss2 = kuhn.play(player1, player2)
        loss1.backward()
        loss2.backward()
    except:
        console = Console()
        console.print_exception()
        pdb.post_mortem()
