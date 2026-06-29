import random
import torch
import numpy as np

from .model import MarioDQN
from .memory import ReplayMemory

import torch.nn as nn
import torch.optim as optim


class MarioAgent:

    def __init__(self, state_shape, num_actions):

        self.num_actions = num_actions

        self.device = torch.device("cpu")

        self.model = MarioDQN(
            input_shape=state_shape,
            num_actions=num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayMemory(capacity=100_000)


    def select_action(self, state, epsilon=0.0):

        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        action = torch.argmax(q_values).item()

        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def can_learn(self, batch_size):
        return len(self.memory) >= batch_size
    
    def learn(self, batch_size):
        if not self.can_learn(batch_size):
            return None

        batch = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        print("States:", states.shape)
        print("Actions:", actions.shape)
        print("Rewards:", rewards.shape)
        print("Next states:", next_states.shape)
        print("Dones:", dones.shape)

        q_values = self.model(states)

        current_q_values = q_values.gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        print("All Q-values:", q_values.shape)
        print("Current Q-values:", current_q_values.shape)
        print("Current Q-values:", current_q_values)

        gamma = 0.9

        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]

        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        print("Next Q-values:", next_q_values.shape)
        print("Max next Q-values:", max_next_q_values.shape)
        print("Target Q-values:", target_q_values.shape)
        print("Target Q-values:", target_q_values)
        

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Loss:", loss.item())

        return loss.item()