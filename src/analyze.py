import torch

def analyze_strategy(player):
    # Analyze as player 1
    print('--- as player 1 ---')
    for history, name in (
        ([0,0,0,0,0,0], 'new game'),
        ([0,0,0,1,0,0],' pass -> bet')):
        history = torch.tensor(history, dtype = torch.float32)
        print(name)
        for card, card_name in (
            [[1,0,0], 'jack'],
            [[0,1,0], 'queen'],
            [[0,0,1], 'king']):
            print(f'holding a {card_name}')
            card = torch.tensor(card, dtype = torch.float32)
            distribution = torch.softmax(player(torch.cat((card, history), dim = 0)), dim = 0)
            print(distribution.tolist())

    # Analysis of player 2
    print('--- as player 2 ---')
    for history, name in (
        ([1,0,0,0,0,0], 'p1 passed'),
        ([0,1,0,0,0,0],' p1 called')):
        history = torch.tensor(history, dtype = torch.float32)
        print(name)
        for card, card_name in (
            [[1,0,0], 'jack'],
            [[0,1,0], 'queen'],
            [[0,0,1], 'king']):
            print(f'holding a {card_name}')
            card = torch.tensor(card, dtype = torch.float32)
            distribution = torch.softmax(player(torch.cat((card, history), dim = 0)), dim = 0)
            print(distribution.tolist())