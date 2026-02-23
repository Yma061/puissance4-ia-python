import torch
from training_ia.model import Connect4Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Connect4Net().to(device)
model.load_state_dict(torch.load("models/connect4_model.pth", map_location=device))
model.eval()


def predict_move(board):
    state = torch.FloatTensor(board).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state)

    return torch.argmax(q_values).item()