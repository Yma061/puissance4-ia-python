import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from training_ia.model import Connect4Net
from training_ia.environment import Connect4SelfPlayEnv
from game.board import is_valid_location

# ===== Hyperparamètres =====
EPISODES = 100000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
LR = 0.0005
BATCH_SIZE = 64
TARGET_UPDATE = 1000
MEMORY_SIZE = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Réseaux =====
policy_net = Connect4Net().to(device)
target_net = Connect4Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

memory = deque(maxlen=MEMORY_SIZE)

env = Connect4SelfPlayEnv()

# ===== Replay Buffer Training =====
def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # ===== Double DQN =====
    next_actions = policy_net(next_states).argmax(1)
    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

    target_q = rewards + GAMMA * next_q * (1 - dones)

    loss = loss_fn(current_q, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


try:
    step_count = 0

    for episode in range(EPISODES):

        state = env.reset()
        done = False

        while not done:

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # ===== ACTION MASKING =====
            valid_moves = [c for c in range(7) if is_valid_location(env.board, c)]

            if random.random() < EPSILON:
                action = random.choice(valid_moves)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()

                masked_q = np.full(7, -np.inf)
                for c in valid_moves:
                    masked_q[c] = q_values[c]

                action = np.argmax(masked_q)

            next_state, reward, done = env.step(action)

            memory.append((state, action, reward, next_state, done))

            state = next_state

            replay()

            step_count += 1

            # ===== Target network update =====
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        if episode % 1000 == 0:
            empty_state = env.reset()
            empty_tensor = torch.tensor(empty_state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = policy_net(empty_tensor).squeeze(0).cpu().numpy()

            print("====================================")
            print(f"Episode {episode}")
            print(f"Epsilon: {EPSILON:.3f}")
            print("Q-values (plateau vide):")
            print(np.round(q_values, 3))
            print("====================================")

    print("Training terminé.")

except KeyboardInterrupt:
    print("Interruption...")

# ===== Sauvegarde =====
import os
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)

torch.save(policy_net.state_dict(), os.path.join(model_dir, "connect4_model.pth"))
print("Modèle sauvegardé.")