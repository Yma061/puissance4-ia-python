import torch
import numpy as np

from training_ia.model import Connect4Net
from game.ai_minimax import minimax
from game.board import *
from game.rules import winning_move


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Connect4Net().to(device)
model.load_state_dict(torch.load("models/connect4_model.pth", map_location=device))
model.eval()


def deep_move(board):
    state = np.copy(board)
    state[state == 2] = -1
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state).squeeze(0).cpu().numpy()

    valid_moves = [c for c in range(7) if is_valid_location(board, c)]
    masked_q = np.full(7, -np.inf)

    for c in valid_moves:
        masked_q[c] = q_values[c]

    return np.argmax(masked_q)


def play_game():

    board = create_board()
    turn = 0  # 0 = Deep, 1 = Minimax

    while True:

        if turn == 0:
            col = deep_move(board)
            piece = 1
        else:
            col, _ = minimax(board, 4, -np.inf, np.inf, True)
            piece = 2

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, piece)

            if winning_move(board, piece):
                return turn  # retourne gagnant

        if len([c for c in range(7) if is_valid_location(board, c)]) == 0:
            return -1  # nul

        turn = 1 - turn


deep_wins = 0
minimax_wins = 0
draws = 0

for i in range(20):

    result = play_game()

    if result == 0:
        deep_wins += 1
    elif result == 1:
        minimax_wins += 1
    else:
        draws += 1

print("Résultats :")
print("Deep IA :", deep_wins)
print("Minimax :", minimax_wins)
print("Nuls :", draws)
# ===== Q-values finales plateau vide =====
empty_board = create_board()
state = np.copy(empty_board)
state[state == 2] = -1
state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

with torch.no_grad():
    final_q = model(state_tensor).squeeze(0).cpu().numpy()

print("\nQ-values finales (plateau vide):")
print(np.round(final_q, 3))