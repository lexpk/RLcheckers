import ctypes
from random import choice, choices
import torch
from torch.utils.data import Dataset

class Position:
    '''
    Implementation of a checkers Board state. This class handles most of the game logic.
    '''
    def __init__(self, squares = [1 for _ in range(12)] + [0 for _ in range(8)] + [-1 for _ in range(12)], color = 1):
        '''
        Initializes new board state.

            Parameters:
                squares: 32 element list encoding the content of each square (default: starting position)
                            1 = dark man, 2 = dark king, -1 = light man, -2 = light king
                color: the side which has the move, 1 for dark, -1 for light (default dark)
        '''
        self.squares = squares
        self.color = color

    def _legal_dark_single_captures_no_promotion(self, piece):
        result = []
        if self.squares[piece] > 0:
            if piece < 24:
                if piece % 4 != 0 and self.squares[piece + 3 + (piece//4)%2] in [-1, -2] and self.squares[piece + 7] == 0:
                    result.append((
                        Position(
                            [0 if i in [piece, piece + 3 + (piece//4)%2] else self.squares[piece] if i == piece + 7 else self.squares[i] for i in range(32)],
                            self.color
                        ), 
                        piece + 7
                    ))
                if piece % 4 != 3 and self.squares[piece + 4 + (piece//4)%2] in [-1, -2] and self.squares[piece + 9] == 0:
                    result.append((
                        Position(
                            [0 if i in [piece, piece + 4 + (piece//4)%2] else self.squares[piece] if i == piece + 9 else self.squares[i] for i in range(32)],
                            self.color
                        ),
                        piece + 9
                    ))
            if self.squares[piece] == 2 and piece >= 8:
                if piece % 4 != 0 and self.squares[piece - 5 + (piece//4)%2] in [-1, -2] and self.squares[piece - 9] == 0:
                    result.append((
                        Position(
                            [0 if i in [piece, piece - 5 + (piece//4)%2] else self.squares[piece] if i == piece - 9 else self.squares[i] for i in range(32)],
                            self.color
                            )
                            , piece - 9
                        ))
                if piece % 4 != 3 and self.squares[piece - 4 + (piece//4)%2] in [-1, -2] and self.squares[piece - 7] == 0:
                    result.append((
                        Position(
                            [0 if i in [piece, piece - 4 + (piece//4)%2] else self.squares[piece] if i == piece - 7 else self.squares[i] for i in range(32)],
                            self.color
                            )
                            , piece - 7
                        ))
        return result

    def _legal_dark_captures_no_promotion(self, single_captures):
        captures = single_captures
        for pos, p in captures:
            captures += pos._legal_dark_single_captures_no_promotion(p)
        return [pos for pos, _ in captures]

    def legal_dark_captures_no_promotion(self):
        result = []
        for piece in range(32):
            result += self._legal_dark_captures_no_promotion(piece, self._legal_dark_single_captures_no_promotion(piece))
        return result

    def legal_dark_non_captures_no_promotion(self):
        result = []
        for piece in range(32):
            if self.squares[piece] > 0:
                if piece < 28:
                    if piece % 8 != 0 and self.squares[piece + 3 + (piece//4)%2] == 0:
                        result.append(
                            Position(
                                [0 if i == piece else self.squares[piece] if i == piece + 3 + (piece//4)%2 else self.squares[i] for i in range(32)],
                                self.color
                            )
                        )
                    if piece % 8 != 7 and self.squares[piece + 4 + (piece//4)%2] == 0:
                        result.append(
                            Position(
                                [0 if i == piece else self.squares[piece] if i == piece + 4 + (piece//4)%2 else self.squares[i] for i in range(32)],
                                self.color
                            )
                        )
                if self.squares[piece] == 2 and piece >= 4:
                    if piece % 8 != 0 and self.squares[piece - 5 + (piece//4)%2] == 0:
                        result.append(
                            Position(
                                [0 if i == piece else 2 if i == piece - 5 + (piece//4)%2 else self.squares[i] for i in range(32)],
                                self.color
                            )
                        )
                    if piece % 8 != 7 and self.squares[piece - 4 + (piece//4)%2] == 0:
                        result.append(
                            Position(
                                [0 if i == piece else 2 if i == piece - 4 + (piece//4)%2 else self.squares[i] for i in range(32)],
                                self.color
                            )
                        )
        return result

    def legal_moves(self):
        '''
        Returns list containing all position that can be moved from given position in a single ply.
        '''
        if self.color == -1:
            positions = [Position.flip(pos) for pos in Position.flip(self).legal_moves()]
            return positions
        single_captures = []
        for piece in range(32):
            single_captures += self._legal_dark_single_captures_no_promotion(piece)
        if single_captures:
            return [
                Position(
                    [2 if i >= 28 and capture.squares[i] == 1 else capture.squares[i] for i in range(32)],
                    -capture.color
                ) for capture in self._legal_dark_captures_no_promotion(single_captures)
            ]
        return [
            Position(
                [2 if i >= 28 and capture.squares[i] == 1 else capture.squares[i] for i in range(32)],
                -capture.color
            ) for capture in self.legal_dark_non_captures_no_promotion()
        ]   
        
    def has_captures(self):
        '''
        Returns true iff there are captures in the position
        '''
        if self.color == -1:
            pos = Position.flip(self)
        else:
            pos = self
        return any(self._legal_dark_single_captures_no_promotion(i) for i in range(32))
    
    def flip(position):
        '''
        Returns the position with colors reversed
        '''
        return Position([-piece for piece in position.squares[::-1]], -position.color)

    def nn_input(self):
        '''
        Transforms a position into a form palattable for the learning by the MCTD agent
        '''
        position = self if self.color == 1 else Position.flip(self)
        return torch.tensor(
            [2 + position.squares[i] + 4 * i for i in range(32)]
        )

    class CPosition(ctypes.Structure):
        _fields_ = [
            ("bm", ctypes.c_uint32),
            ("bk", ctypes.c_uint32),
            ("wm", ctypes.c_uint32),
            ("wk", ctypes.c_uint32),
        ]

    def c_position(self):
        '''
        Transforms a position into a form palattable for the C engine
        '''
        c_input = Position.CPosition()
        for i in range(32):
            if self.squares[i] == 1:
                c_input.bm |= 1 << i
            if self.squares[i] == 2:
                c_input.bk |= 1 << i
            if self.squares[i] == -1:
                c_input.wm |= 1 << i
            if self.squares[i] == -2:
                c_input.wk |= 1 << i
        return c_input
        
    def from_cposition(c_position, color):
        '''
        Transforms a C position into a Python position
        '''
        squares = []
        for i in range(32):
            if c_position.bm & (1 << i):
                squares.append(1)
            elif c_position.bk & (1 << i):
                squares.append(2)
            elif c_position.wm & (1 << i):
                squares.append(-1)
            elif c_position.wk & (1 << i):
                squares.append(-2)
            else:
                squares.append(0)
        return Position(squares, color)


    def random(light_men, light_kings, dark_men, dark_kings):
        '''
        Returns a random position with specified material:

            Parameters:
                light_men: number of light men
                light_kings: number of light kings
                dark_men: number of dark men
                dark_kings: number of dark kings
        '''
        result = [0 for _ in range(32)]
        pieces = choices(list(range(28)), k=light_men)
        result = [1 if i in pieces else result[i] for i in range(32)]
        pieces = choices([i for i in range(32) if result[i] == 0], k=light_kings)
        result = [2 if i in pieces else result[i] for i in range(32)]
        pieces = choices([i for i in range(4, 32) if result[i] == 0], k=dark_men)
        result = [-1 if i in pieces else result[i] for i in range(32)]
        pieces = choices([i for i in range(32) if result[i] == 0], k=dark_kings)
        result = [-2 if i in pieces else result[i] for i in range(32)]
        return Position(result, choice([-1, 1]))

    def __eq__(self, other):
        return self.squares == other.squares and self.color == other.color

    def __hash__(self):
        return sum([abs(self.squares[i])*(2**i) for i in range(32)])

    def ascii(self):
        '''
        Returns ascii representation of the position
        '''
        result = ""
        for i in range(32):
            if i % 4 == 0:
                result += "\n"
            if self.squares[i] == 1:
                result += " b    " if i // 4 % 2 == 0 else "    b "
            if self.squares[i] == 2:
                result += " B    " if i // 4 % 2 == 0 else "    B "
            if self.squares[i] == -1:
                result += " w    " if i // 4 % 2 == 0 else "    w "
            if self.squares[i] == -2:
                result += " W    " if i // 4 % 2 == 0 else "    W "
            if self.squares[i] == 0:
                result += " .    " if i // 4 % 2 == 0 else "    . "
        return result[::-1]


class PositionDataset(Dataset):
    '''
    Dataset for the MCTD agent. It generates random positions and their evaluations.
    '''
    def __init__(self, positions, evaluations):
        '''
        Initializes the dataset.

            Parameters:
                positions: tensor of positions
                evaluations: tensor of evaluations
        '''
        self.positions = positions
        self.evaluations = evaluations

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

    def save(self, path):
        '''
        Saves the dataset to a file.

            Parameters:
                path: path to the file
        '''
        torch.save((self.positions, self.evaluations), "./data/" + path)

    def load(path):
        '''
        Loads the dataset from a file.

            Parameters:
                path: path to the file
        '''
        positions, evaluations = torch.load("./data/" + path, weights_only=True)
        return PositionDataset(positions, evaluations)
