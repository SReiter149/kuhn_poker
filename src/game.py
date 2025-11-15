from torch import randperm
import torch
import pdb

class KuhnPoker():
    def __init__(self,):
        self.device = 'cpu'
        self.game_state = []

    def get_game_state(self):
        return_state = self.game_state + [1,] * (2 - len(self.game_state))
        return torch.asarray(return_state, device = self.device, dtype = torch.float32)

    def deal(self):
        self.p1_card, self.p2_card, _ = randperm(3)
        

    def move(self, move):
        """
        move = <0 represents pass
        move = >0 represents bet
        """
        if move < 0:
            self.game_state.append(0)
        else:
            self.game_state.append(1)

    def showdown(self):
        if self.p1_card > self.p2_card:
            return torch.asarray((1,-1), dtype = torch.int32, device = self.device)
        return torch.asarray((-1,1), dtype = torch.int32, device = self.device)

    def payout(self):
        """
        
        """
        if self.game_state == [0,0]:
            return self.showdown()
        elif self.game_state[-2: ] == [1,1]:
            return 2 * self.showdown()
        elif self.game_state == [0,1,0]:
            return torch.asarray((-1,1), dtype = torch.int32, device = self.device)
        elif self.game_state == [1,0]:
            return torch.asarray((1,-1), dtype = torch.int32, device = self.device)
        else:
            pdb.set_trace()

        
    def play(self, policy1, policy2):
        self.deal()

        move1 = policy1(self.get_game_state())
        self.move(move1)

        move2 = policy2(self.get_game_state())
        self.move(move2)

        if self.game_state == [0,1]:
            print("here")
            move3 = policy1(self.get_game_state())
            self.move(move3)
        
        payout = self.payout()

        print(f"the cards were {(self.p1_card.item(), self.p2_card.item())}")
        print(f'the game state was {self.game_state}')
        print(f'payout is {payout.tolist()}')
        return payout

if __name__ == "__main__":
    kuhn = KuhnPoker()
    kuhn.play()



    

