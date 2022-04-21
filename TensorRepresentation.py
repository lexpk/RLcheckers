from typing import List
import torch



class TensorRepresentation():
    '''
    A Class that keeps track of the Checkers rules.
    '''
    initialized = False
    
    @classmethod
    def initialize(cls, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        cls.device = device
        cls.STARTING_POSITION = torch.tensor(
            [
                [i < 12 for i in range(32)],            # squares occupied by dark men
                [False for _ in range(32)],             # squares occupied by dark kings
                [i < 12 for i in range(32)],            # squares occupied by dark pieces
                [i >= 20 for i in range(32)],           # squares occupied by light men
                [False for _ in range(32)],             # squares occupied by light kings
                [i >= 20 for i in range(32)],           # squares occupied by light pieces
            ],
            dtype=torch.float,
            device=cls.device
        )
        cls.EXAMPLE_MAN_CAPTURE_POSITION = torch.tensor(
            [
                [i == 1 for i in range(32)],                      # squares occupied by dark men
                [False for _ in range(32)],                       # squares occupied by dark kings
                [i == 1 for i in range(32)],                      # squares occupied by dark pieces
                [i in [4, 12, 21, 26]  for i in range(32)],       # squares occupied by light men
                [False for _ in range(32)],                       # squares occupied by light kings
                [i in [4, 12, 21, 26] for i in range(32)],        # squares occupied by light pieces
            ],
            dtype=torch.float,
            device=cls.device
        )
        cls.LONG_KING_POSITION = torch.tensor(
            [
                [False for _ in range(32)],                               # squares occupied by dark men
                [i == 17 for i in range(32)],                             # squares occupied by dark kings
                [False for _ in range(32)],                               # squares occupied by dark pieces
                [i in [5, 12, 13, 22]  for i in range(32)],               # squares occupied by light men
                [i in [4, 21] for i in range(32)],                        # squares occupied by light kings
                [i in [4, 5, 12, 13, 21, 22]  for i in range(32)],               # squares occupied by light pieces
            ],
            dtype=torch.float,
            device=cls.device
        )
        single_man_captures = [          
            [
                [j == i for j in range(32)],                    # squares that need to be occupied by a dark men / initial square
                [False for _ in range(32)],                     # squares that need to be occupied by a dark king / inital square
                [j == i + 4 + (i//4)%2 for j in range(32)],     # squares that need to be occupied by a white pieces / captured stones
                [j == i or j == i + 9 for j in range(32)],      # square that need to be free / path that the stone takes
                [j == i + 9 and i < 20 for j in range(32)],     # squares that dark men end up at / destination
                [j == i + 9 and i >= 20 for j in range(32)],    # squares that dark kings end up at / destination
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [j == i for j in range(32)],
                [False for _ in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [j == i or j == i + 7 for j in range(32)],
                [j == i + 7 and i < 20 for j in range(32)],
                [j == i + 7 and i >= 20 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ]
        man_captures = single_man_captures
        composable_captures = single_man_captures.copy()
        for capture in composable_captures:
            for extension in single_man_captures:
                if capture[4] == extension[0] and not any(x and y for x, y in zip(capture[2], extension[2])):
                    new_capture = [
                        capture[0],
                        [False for _ in range(32)],
                        [x or y for x, y in zip(capture[2], extension[2])],
                        [x or y for x, y in zip(capture[3], extension[3])],
                        extension[4],
                        extension[5]
                    ]
                    if new_capture not in man_captures: 
                        man_captures.append(new_capture)
                        composable_captures.append(new_capture)
       
        single_king_captures =  [
            [
                [False for _ in range(32)],
                [j == i for j in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
                [j == i or j == i + 9 for j in range(32)],
                [False for _ in range(32)],
                [j == i + 9 for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [False for _ in range(32)],
                [j == i for j in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [j == i or j == i + 7 for j in range(32)],
                [False for _ in range(32)],
                [j == i + 7 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ] + [
            [
                [False for _ in range(32)],
                [j == i + 9 for j in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
                [j == i + 9 or j == i for j in range(32)],
                [False for _ in range(32)],
                [j == i for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [False for _ in range(32)],
                [j == i + 7 for j in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [j == i + 7 or j == i for j in range(32)],
                [False for _ in range(32)],
                [j == i for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ]
        king_captures = single_king_captures
        composable_captures = single_king_captures.copy()
        for capture in composable_captures:
            for extension in single_king_captures:
                if capture[-1] == extension[1] and not any(x and y for x, y in zip(capture[2], extension[2])):
                    new_capture = [
                        [False for _ in range(32)],
                        capture[1],
                        [x or y for x, y in zip(capture[2], extension[2])],
                        [x or y for x, y in zip(capture[3], extension[3])],
                        [False for _ in range(32)],
                        extension[-1]
                    ]
                    if new_capture not in king_captures:
                        king_captures.append(new_capture)
                        composable_captures.append(new_capture)
        
        non_captures = [
            [
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [False  for _ in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
                [j == i + 4 + (i//4)%2 and i < 24 for j in range(32)],
                [j == i + 4 + (i//4)%2 and i >= 24 for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 28
        ] + [
            [
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [False  for _ in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [j == i + 3 + (i//4)%2 and i < 24 for j in range(32)],
                [j == i + 3 + (i//4)%2 and i >= 24 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 28
        ] + [
            [
                [False  for _ in range(32)],
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
                [False  for _ in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 28
        ] + [
            [
                [False  for _ in range(32)],
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [False  for _ in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 28
        ] + [
            [
                [False  for _ in range(32)],
                [j == i + 4 + (i//4)%2 for j in range(32)],
                [False  for _ in range(32)],
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [j == i for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 28
        ] + [
            [
                [False  for _ in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [False  for _ in range(32)],
                [j == i for j in range(32)],
                [False  for _ in range(32)],
                [j == i for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 28
        ]

        moves = man_captures + king_captures + non_captures
        cls.CAPTURE_COUNT = len(man_captures + king_captures)
        padding = torch.zeros((len(moves), 32), dtype=torch.float, device=cls.device)
        cls._SUFFICIENCY_FILTER = (1 - torch.stack((
            torch.tensor([c[0] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[1] for c in moves], dtype=torch.float, device=cls.device),
            padding,
            padding,
            padding,
            torch.tensor([c[2] for c in moves], dtype=torch.float, device=cls.device),
        ), dim=1)).clone().detach()
        cls._FREENESS_FILTER = torch.stack((
            padding,
            padding,
            torch.tensor([[x and not y and not z for x, y, z in zip(c[3], c[0], c[1])] for c in moves], dtype=torch.float, device=cls.device),
            padding,
            padding,
            torch.tensor([[x and not y and not z for x, y, z in zip(c[3], c[0], c[1])] for c in moves], dtype=torch.float, device=cls.device),
        ), dim=1).clone().detach()
        cls._REMOVER = (1 - torch.stack((
            torch.tensor([c[0] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[1] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([[x or y for x, y in zip(c[0], c[1])] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[2] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[2] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[2] for c in moves], dtype=torch.float, device=cls.device),
        ), dim=1)).clone().detach()
        cls._ADDER = torch.stack((
            torch.tensor([c[4] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([c[5] for c in moves], dtype=torch.float, device=cls.device),
            torch.tensor([[x or y for x, y in zip(c[4], c[5])] for c in moves], dtype=torch.float, device=cls.device),
            padding,
            padding,
            padding,
        ), dim=1).clone().detach()
        cls.MOVES = torch.tensor([c[3] for c in moves], dtype=torch.float, device=cls.device)

        cls.initialized = True


    def next_positions(positions : torch.tensor) -> List[torch.tensor]:
        expanded_positions = positions.expand(len(TensorRepresentation.MOVES), len(positions), 6, 32).transpose(0, 1)
        mask = (
                torch.maximum(expanded_positions, TensorRepresentation._SUFFICIENCY_FILTER.expand(len(positions), len(TensorRepresentation.MOVES), 6, 32))
            ).amin([2, 3]) * (
                1 - (expanded_positions * TensorRepresentation._FREENESS_FILTER.expand(len(positions), len(TensorRepresentation.MOVES), 6, 32)).amax([2, 3])
            ).flatten(1)
        indices = [i[i < TensorRepresentation.CAPTURE_COUNT] if i[0] < TensorRepresentation.CAPTURE_COUNT else i[i >= TensorRepresentation.CAPTURE_COUNT] for i in [layer.argwhere() for layer in mask]]
        next = torch.maximum(
            expanded_positions * TensorRepresentation._REMOVER.expand(len(positions), len(TensorRepresentation.MOVES), 6, 32),
            TensorRepresentation._ADDER.expand(len(positions), len(TensorRepresentation.MOVES), 6, 32)
        )
        return [options[i][:, [3, 4, 5, 0, 1, 2]].flip(dims=(2,)) for options, i in zip(next, indices)]
    
    def to_32(position : torch.tensor):
        return [int(position[0][i]) + 2*int(position[1][i]) -int(position[3][i]) - 2*int(position[4][i]) for i in range(32)]

    def from_32(position, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        return torch.tensor(
            [
                [position[i] == 1 for i in range(32)],
                [position[i] == 2 for i in range(32)],
                [position[i] in [1, 2] for i in range(32)],
                [position[i] == -1 for i in range(32)],
                [position[i] == -2 for i in range(32)],
                [position[i] in [-1, -2] for i in range(32)],
            ],
            dtype=torch.float,
            device=device
        )

