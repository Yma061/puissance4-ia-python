import math
import random
from game.board import get_next_open_row, drop_piece, is_valid_location
from game.rules import winning_move

AI_PIECE = 2
PLAYER_PIECE = 1

def get_valid_locations(board):
    return [col for col in range(7) if is_valid_location(board, col)]

def minimax(board, depth, alpha, beta, maximizingPlayer):

    valid_locations = get_valid_locations(board)

    if depth == 0:
        return random.choice(valid_locations), 0

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, AI_PIECE)

            new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                column = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return column, value

    else:
        value = math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, PLAYER_PIECE)

            new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                column = col

            beta = min(beta, value)
            if alpha >= beta:
                break

        return column, value