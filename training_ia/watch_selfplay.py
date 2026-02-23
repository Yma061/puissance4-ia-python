import torch
import numpy as np
import time

from training_ia.model import Connect4Net
from training_ia.environment import Connect4SelfPlayEnv
from game.board import is_valid_location

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Connect4Net().to(device)
model.load_state_dict(torch.load("models/connect4_model.pth", map_location=device))
model.eval()

env = Connect4SelfPlayEnv()

state = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
done = False


def print_board(board):
    print(np.flip(board, 0))
    print("\n" + "-" * 30 + "\n")


print("🎮 Self-play Deep vs Deep\n")

while not done:

    with torch.no_grad():
        q_values = model(state).squeeze(0).cpu().numpy()

    # 🔥 Masquage des colonnes invalides
    valid_moves = [c for c in range(7) if is_valid_location(env.board, c)]

    masked_q = np.full(7, -np.inf)
    for c in valid_moves:
        masked_q[c] = q_values[c]

    action = np.argmax(masked_q)

    next_state_np, reward, done = env.step(action)

    print_board(env.board)
    time.sleep(0.7)

    state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

print("Fin de la partie.")