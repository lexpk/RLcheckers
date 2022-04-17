from turtle import pos
import torch

class Rules():
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
                [i < 12 for i in range(32)],            # squares occupied by dark pieves
                [i >= 20 for i in range(32)],           # squares occupied by light men
                [False for _ in range(32)],             # squares occupied by light kings
                [i >= 20 for i in range(32)],           # squares occupied by light pieces
                [i >= 12 and i < 20 for i in range(32)] # empty squares
            ],
            dtype=torch.bool,
            device=device
        )
        cls.EXAMPLE_MAN_CAPTURE_POSITION = torch.tensor(
            [
                [i == 1 for i in range(32)],                      # squares occupied by dark men
                [False for _ in range(32)],                       # squares occupied by dark kings
                [i == 1 for i in range(32)],                      # squares occupied by dark pieves
                [i in [4, 12, 21, 25]  for i in range(32)],       # squares occupied by light men
                [False for _ in range(32)],                       # squares occupied by light kings
                [i in [4, 12, 21, 26] for i in range(32)],        # squares occupied by light pieces
                [i not in [1, 4, 12, 21, 26] for i in range(32)], # empty squares
            ],
            dtype=torch.bool,
            device=device
        )
        cls.LONG_KING_CAPTURE_POSITION = torch.tensor(
            [
                [False for _ in range(32)],                               # squares occupied by dark men
                [i == 17 for i in range(32)],                             # squares occupied by dark kings
                [i == 17 for i in range(32)],                             # squares occupied by dark pieves
                [i in [12, 5, 13, 22]  for i in range(32)],               # squares occupied by light men
                [i in [4, 21] for i in range(32)],                        # squares occupied by light kings
                [i in [4, 5, 12, 13, 21, 22] for i in range(32)],         # squares occupied by light pieces
                [i not in [4, 5, 12, 13, 17, 21, 22] for i in range(32)], # empty squares
            ],
            dtype=torch.bool,
            device=device
        )
        single_man_captures = [          
            [
                [j != i for j in range(32)],
                [True for _ in range(32)],
                [j != i + 4 + (i//4)%2 for j in range(32)],
                [j != i + 9 or i >= 20 for j in range(32)],
                [j != i + 9 or i < 20 for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [j != i for j in range(32)],
                [True for _ in range(32)],
                [j != i + 3 + (i//4)%2 for j in range(32)],
                [j != i + 7 or i >= 20 for j in range(32)],
                [j != i + 7 or i < 20 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ]
        man_captures = single_man_captures
        composable_captures = single_man_captures.copy()
        for capture in composable_captures:
            for extension in single_man_captures:
                if capture[-2] == extension[0] and all(x or y for x, y in zip(capture[2], extension[2])):
                    new_capture = [
                        capture[0],
                        [True for _ in range(32)],
                        [x and y for x, y in zip(capture[2], extension[2])],
                        extension[-2],
                        extension[-1]
                    ]
                    if new_capture not in man_captures:
                        man_captures.append(new_capture)
                        composable_captures.append(new_capture)
       
        single_king_captures =  [
            [
                [True for _ in range(32)],
                [j != i for j in range(32)],
                [j != i + 4 + (i//4)%2 for j in range(32)],
                [True for _ in range(32)],
                [j != i + 9 for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [True for _ in range(32)],
                [j != i for j in range(32)],
                [j != i + 3 + (i//4)%2 for j in range(32)],
                [True for _ in range(32)],
                [j != i + 7 for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ] + [
            [
                [True for _ in range(32)],
                [j != i + 9 for j in range(32)],
                [j != i + 4 + (i//4)%2 for j in range(32)],
                [True for _ in range(32)],
                [j != i for j in range(32)],
            ] for i in range(32) if i%4 != 3 and i < 24
        ] + [
            [
                [True for _ in range(32)],
                [j != i + 7 for j in range(32)],
                [j != i + 3 + (i//4)%2 for j in range(32)],
                [True for _ in range(32)],
                [j != i for j in range(32)],
            ] for i in range(32) if i%4 != 0 and i < 24
        ]
        king_captures = single_king_captures
        composable_captures = single_king_captures.copy()
        cycle_correction = [[False for _ in range(32)] for _ in range(len(man_captures) + len(king_captures))]
        for capture in composable_captures:
            for extension in single_king_captures:
                if capture[-1] == extension[1] and all(x or y for x, y in zip(capture[2], extension[2])):
                    new_capture = [
                        [True for _ in range(32)],
                        capture[1],
                        [x and y for x, y in zip(capture[2], extension[2])],
                        [True for _ in range(32)],
                        extension[-1]
                    ]
                    if new_capture not in king_captures:
                        king_captures.append(new_capture)
                        composable_captures.append(new_capture)
                        cycle_correction.append(
                            [not b for b in new_capture[1]] if new_capture[1] == new_capture[-1] else [False for _ in range(32)]
                        )
        
        cls.CAPTURES = torch.tensor(man_captures + king_captures, dtype=torch.bool, device=device)
        padding = torch.zeros(size=(len(cycle_correction), 32), dtype = torch.bool, device=device)
        cls.CYCLE_CORRECTION = torch.stack((padding, padding, padding, padding, torch.tensor(cycle_correction, dtype=torch.bool, device=device))).transpose(0, 1)
        cls.NON_CAPTURES = torch.tensor(
            [
                [
                    [j != i for j in range(32)],
                    [1  for _ in range(32)],
                    [j != i + 4 + (i//4)%2 for j in range(32)],
                ] for i in range(32) if i%4 != 3 and i < 28
            ] + [
                [
                    [j != i for j in range(32)],
                    [1  for _ in range(32)],
                    [j != i + 3 + (i//4)%2 for j in range(32)],
                ] for i in range(32) if i%4 != 0 and i < 28
            ] + [
                [
                    [1  for _ in range(32)],
                    [j != i for j in range(32)],
                    [j != i + 4 + (i//4)%2 for j in range(32)],
                ] for i in range(32) if i%4 != 3 and i < 28
            ] + [
                [
                    [1  for _ in range(32)],
                    [j != i for j in range(32)],
                    [j != i + 3 + (i//4)%2 for j in range(32)],
                ] for i in range(32) if i%4 != 0 and i < 28
            ] + [
                [
                    [1  for _ in range(32)],
                    [j != i + 4 + (i//4)%2 for j in range(32)],
                    [j != i for j in range(32)],
                ] for i in range(32) if i%4 != 3 and i < 28
            ] + [
                [
                    [1  for _ in range(32)],
                    [j != i + 3 + (i//4)%2 for j in range(32)],
                    [j != i for j in range(32)],
                ] for i in range(32) if i%4 != 0 and i < 28
            ],
            dtype=torch.bool,
            device=device
        )

        cls.initialized = True


    def legal_captures(position : torch.tensor):
        pos = position[[0, 1, 5, 6, 6]].expand(len(Rules.CAPTURES), 5, 32)
        mask = pos.bitwise_or(Rules.CYCLE_CORRECTION).bitwise_or(Rules.CAPTURES).amin([1, 2])
        return Rules.CAPTURES[mask.argwhere()]

    def legal_non_captures(position : torch.tensor):
        pos = position[[0, 1, 5, 6, 6]].expand(len(Rules.NON_CAPTURES), 5, 32)
        mask = pos.bitwise_or(Rules.NON_CAPTURES).amin([1, 2])
        return Rules.NON_CAPTURES[mask.argwhere()]

    def legal_moves(position):
        captures = Rules.legal_captures(position)
        return captures if captures else Rules.legal_non_captures(position)