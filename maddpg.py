import numpy as np
import random
import copy
from collections import namedtuple, deque

from configuration import Configuration
from ddpg import DDPGAgent

import torch
import torch.nn.functional as F
import torch.optim as optim



config = Configuration()


class MADDPGAgent():
    """A class to create MADDPG agents, to train in the enviroment and compete/collabrate."""
    def __init__(self, state_size, action_size, n_agents, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(seed)
        
        self.time_step = 0
        
        # Create agents
        self.agents = [DDPGAgent(state_size, action_size, index, seed) for index in range(self.n_agents)]
        
        # Noise process for exploratary action
        self.noise = OUNoise(action_size, self.seed)
    
        # Set replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, self.seed)
        
        self.epsilon = config.epsilon
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and use a random batch from memory to learn."""
        
        states = states.reshape(2, -1) 
        next_states = next_states.reshape(2, -1)
        actions = actions.reshape(2, -1)
        #rewards = rewards.reshape(1, -1)
        #dones = dones.reshape(1, -1)
        
        for i in range(self.n_agents):
            self.memory.add(states[i,:], actions[i,:], rewards[i], next_states[i,:], dones[i])

        #self.memory.add(states, actions, rewards, next_states, dones)    
        # Update time step
        self.time_step += 1
        
        # If enough samples are availble in the buffer to sample, Learn
        if len(self.memory) > config.batch_size and self.time_step % config.update_every == 0:
            for i in range(config.num_update):
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                #experiences = self.memory.sample()
                self.learn(experiences, config.gamma)
                
                
    def learn(self, experiences, gamma):
        """LEARN!!"""
        for i, agent in enumerate(self.agents):
            agent.learn(experiences[i], gamma)
            # Reset noise                               
            self.reset()  
            
        # Decay epsilon
        self.epsilon -= config.epsilon_decay       
                
    def act(self, states, add_noise=True):
        """Act!!!!"""
        actions = []
        
        for agent, state in zip(self.agents, states):
            action = agent.act(state)
            
            if add_noise:
                action += self.epsilon * self.noise.sample()
                #self.reset()
            
            action = np.clip(action, -1, 1)
            
            actions.append(action)
            
        return np.array(actions).reshape(1, -1)       
    
    def reset(self):
        self.noise.reset()
        
    def save(self):
        """SAVE!!!"""
        for agent in self.agents:
            self.agent.save()  
         
        
class OUNoise():
    """Ornstein-Uhlenbeck noise process."""
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=1):
        """Initialize parameters and nose process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        """Reset the internal state to mean."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        
        return self.state
    
    
class ReplayBuffer():
    """Replay buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params:
        buffer_size: maximum size of buffer
        batch_size: size of each training batch
        """
        self.action_size = action_size
        # Internal memory
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(config.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(config.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(config.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(config.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(config.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the size of the internal memory."""
        return len(self.memory)