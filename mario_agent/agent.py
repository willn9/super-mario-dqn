import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import MarioDQN
from .memory import ReplayMemory

from config import GAMMA, LEARNING_RATE


class MarioAgent:

    def __init__(self, state_shape, num_actions):

        self.num_actions = num_actions

        self.device = torch.device(
              "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Using device: {self.device}")

        self.model = MarioDQN(
            input_shape=state_shape,
            num_actions=num_actions
        ).to(self.device)


        self.target_model = MarioDQN(
            input_shape=state_shape,
            num_actions=num_actions
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()


        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
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


        q_values = self.model(states)

        current_q_values = q_values.gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]

        target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)


        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item()
    
    def save(self, filename):
        torch.save(
            self.model.state_dict(),
            filename
        )

    def load(self, filename):
        self.model.load_state_dict(
            torch.load(filename, map_location=self.device)
        )

    def update_target_model(self):
        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        
      