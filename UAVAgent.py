import torch
from DQNModel import DQN
import numpy as np
import random
from collections import deque,namedtuple

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class UAVAgent:
    
    def __init__(self):
        self.model = DQN()  # Ana model
        self.target_model = DQN()  # Hedef model
        self.state_size = 3
        self.action_size = 3
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = ReplayMemory(capacity=10000)
        self.gamma = 0.0  # Discount factor
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state','done'))
        self.criterion = torch.nn.MSELoss()
        self.target_update_interval = 10
        
    def findAction(self,state):
        # if np.random.rand() <= self.epsilon:
        #     print("Rastgele bir aksiyon seç")
        #     return torch.tensor(np.random.uniform(-1, 1, size=(1, self.action_size)), dtype=torch.float32)
        # else:
        state = torch.FloatTensor(state).unsqueeze(0)
        actions = self.model(state)
        return actions

    def remember(self,state,action,reward,next_state,done):
        self.memory.push(self.Transition(state,action,reward,next_state,done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))
        state_batch = torch.stack([torch.tensor(state, dtype=torch.float32).unsqueeze(0) for state in batch.state])
        next_state_batch = torch.stack([torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) for next_state in batch.next_state])
        reward_batch = torch.stack([torch.tensor(state, dtype=torch.float32).unsqueeze(1) for state in batch.reward])
        done_batch = torch.stack([torch.tensor(state, dtype=torch.float32) for state in batch.done])
    
        Q_values = self.model(state_batch)

        next_Q_values = self.target_model(next_state_batch).detach()
        # print(done_batch.size())
        # print(next_Q_values.size())
        # next_Q_values *= 1 - done_batch.view(-1, 1)

        # Bellman denklemi ile Q değerlerini güncelle
        expected_Q_values = reward_batch + self.gamma * next_Q_values
        # next_Q_values = (reward_batch + self.gamma*next_Q_values)*self.lr + (1-self.lr)*Q_values
        loss = self.criterion(Q_values, expected_Q_values)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        
        # if self.current_step % self.target_update_interval == 0:
        self.update_target_model()
        # torch.save(self.model.state_dict(), 'model.pth')   
        # torch.save(self.target_model.state_dict(), 'target_model.pth')
        return loss.item()
    
    def update_target_model(self):
        # Target modeli güncelle (Ana modelin parametrelerini target modele kopyala)
        self.target_model.load_state_dict(self.model.state_dict())