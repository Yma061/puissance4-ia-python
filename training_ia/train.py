import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from training_ia.model import Connect4Net
from training_ia.environment import Connect4SelfPlayEnv


EPISODES = 5000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Connect4Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

env = Connect4SelfPlayEnv()

for episode in range(EPISODES):

    state = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)
    done = False

    last_state = None
    last_action = None

    while not done:

        if random.random() < EPSILON:
            action = random.randint(0, 6)
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        next_state_np, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)

        # Si ce n’est pas le premier coup
        if last_state is not None:

            # Si le joueur courant gagne
            if reward == 1:
                target = -1  # L'ancien joueur perd
            else:
                target = 0

            output = model(last_state)[0][last_action]
            target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
            loss = loss_fn(output, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Mise à jour normale du joueur courant
        target = reward
        if not done:
            with torch.no_grad():
                target += GAMMA * torch.max(model(next_state)).item()

        output = model(state)[0][action]
        target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
        loss = loss_fn(output, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_state = state
        last_action = action
        state = next_state

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    if episode % 100 == 0:
        print(f"Episode {episode}")

import os

model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "connect4_model.pth")
torch.save(model.state_dict(), model_path)
print("Self-play training terminé.")