import numpy as np
from game.board import *
from game.rules import winning_move


class Connect4SelfPlayEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = create_board()
        self.current_player = 1  # 1 ou -1
        self.done = False
        return self.get_state()

    def get_state(self):
        # perspective du joueur courant
        state = np.copy(self.board)
        state[state == 2] = -1
        return state * self.current_player

    def step(self, action):

        if not is_valid_location(self.board, action):
            return self.get_state(), -1, True

        piece = 1 if self.current_player == 1 else 2
        row = get_next_open_row(self.board, action)
        drop_piece(self.board, row, action, piece)

        if winning_move(self.board, piece):
            return self.get_state(), 1, True

        if len([c for c in range(COLUMN_COUNT) if is_valid_location(self.board, c)]) == 0:
            return self.get_state(), -0.05, True

        self.current_player *= -1
        return self.get_state(), 0, False