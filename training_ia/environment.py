import numpy as np
from game.board import *
from game.rules import winning_move


class Connect4Env:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = create_board()
        self.done = False
        return self.board

    def step(self, action, piece):

        if not is_valid_location(self.board, action):
            return self.board, -0.5, True  # punition coup illégal

        row = get_next_open_row(self.board, action)
        drop_piece(self.board, row, action, piece)

        if winning_move(self.board, piece):
            return self.board, 1, True

        if len([c for c in range(COLUMN_COUNT) if is_valid_location(self.board, c)]) == 0:
            return self.board, 0, True  # match nul

        return self.board, 0, False