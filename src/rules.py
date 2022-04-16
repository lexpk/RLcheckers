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
                [i < 12 for i in range(32)],
                [0 for _ in range(32)],
                [i >= 20 for i in range(32)],
                [0 for _ in range(32)]
            ],
            device=device
        )
        cls.SINGLE_MAN_CAPTURES = [
            torch.tensor(
                [
                    [j == i for j in range(32)],
                    [j == i + 9 for j in range(32)],
                    [j == i + 4 + (i//4)%2 for j in range(32)],
                ],
            device=device
            ) for i in range(24) if i%4 != 3 and i < 24
        ] + [
            torch.tensor(
            [
                [j == i for j in range(32)],
                [j == i + 7 for j in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
            ],
            device=device
        ) for i in range(32) if i%4 != 0 and i < 24]
        cls.SINGLE_KING_CAPTURES = cls.SINGLE_MAN_CAPTURES + [c[[1,0,2]] for c in cls.SINGLE_MAN_CAPTURES]
        cls.MAN_NON_CAPTURES = [
            torch.tensor(
                [
                    [j == i for j in range(32)],
                    [j == i + 4 + (i//4)%2 for j in range(32)],
                    [0  for _ in range(32)],
                ],
            device=device
            ) for i in range(24) if i%4 != 3 and i < 28
        ] + [
            torch.tensor(
            [
                [j == i for j in range(32)],
                [j == i + 3 + (i//4)%2 for j in range(32)],
                [0  for _ in range(32)],
            ],
            device=device
        ) for i in range(32) if i%4 != 0 and i < 28]

        cls.KING_NON_CAPTURES = cls.MAN_NON_CAPTURES + [c[[1,0,2]] for c in cls.MAN_NON_CAPTURES]

        cls.initialized = True

    def composed_man_captures(c1 : torch.tensor):
        assert Rules.initialized, "Rules has not been initialized. Please call 'Rules.initialize()'"
        result = []
        for c2 in Rules.SINGLE_MAN_CAPTURES:
            if c1[1].bitwise_and(c2[0]).any() and not c1[2].bitwise_and(c2[2]).any():
                result.append(
                    torch.stack(
                        (c1[0], c1[1], c1[2].bitwise_or(c2[2]))
                    )
                )
        return result

    def composed_king_captures(c1 : torch.tensor):
        assert Rules.initialized, "Rules has not been initialized. Please call 'Rules.initialize()'"
        result = []
        for c2 in Rules.SINGLE_KING_CAPTURES:
            if c1[1].bitwise_and(c2[0]).any() and not c1[2].bitwise_and(c2[2]).any():
                result.append(
                    torch.stack(
                        (c1[0], c1[1], c1[2].bitwise_or(c2[2]))
                    )
                )
        return result

    def is_possible(position : torch.tensor, move : torch.tensor):
        return move[0].bitwise_and(position[0]).any() and not move[2].bitwise_and(position[2].bitwise_or(position[3])).any() and (
                not move[1].bitwise_and(position[0].bitwise_or(position[1]).bitwise_or(position[0]).bitwise_or(position[3])).any())

    def legal_captures(position : torch.tensor):
        result = []
        composable = []
        for capture in Rules.SINGLE_MAN_CAPTURES:
            if Rules.is_possible(position, capture):
                result.append(capture)
                composable.append(capture)
        for previous_capture in composable:
            for capture in Rules.composed_man_captures(previous_capture):
                if Rules.is_possible(position, capture):
                    result.append(capture)
                    composable.append(capture)
        for capture in Rules.SINGLE_KING_CAPTURES:
            if Rules.is_possible(position, capture):
                result.append(capture)
                composable.append(capture)
        for previous_capture in composable:
            for capture in Rules.composed_king_captures(previous_capture):
                if Rules.is_possible(position, capture):
                    result.append(capture)
                    composable.append(capture)
        return result

    def legal_non_captures(position : torch.tensor):
        result = []
        for move in Rules.MAN_NON_CAPTURES:
            if Rules.is_possible(position, move):
                result.append(move)
        return result

    