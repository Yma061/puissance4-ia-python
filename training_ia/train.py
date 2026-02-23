import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from model import Connect4Net
from environment import Connect4Env


EPISODES = 2000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Connect4Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

env = Connect4Env()

for episode in range(EPISODES):

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    done = False

    while not done:

        # Exploration / Exploitation
        if random.random() < EPSILON:
            action = random.randint(0, 6)
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action, 2)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        target = reward
        if not done:
            with torch.no_grad():
                target += GAMMA * torch.max(model(next_state_tensor)).item()

        output = model(state)[0][action]
        loss = loss_fn(output, torch.tensor(target).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state_tensor

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    if episode % 100 == 0:
        print(f"Episode {episode}")

torch.save(model.state_dict(), "../models/connect4_model.pth")
print("Entraînement terminé.")