import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from setting import Settings

import torch
import torch.nn.functional as F
import torch.optim as optim


config = Settings()


class DDPGAgent():
    """A class to create DDPG agents that interact and learn from the enviroment."""
    
    def __init__(self, state_size, action_size, n_agents, seed):
        """Initilize the Agent.
        
        Params:
        state_size: dimension of the state
        action_size: dimension of the action
        seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(seed)
        self.epsilon = config.epsilon
        
        # Time step number
        self.time_step = 0
        
        # Set up the Actor networks
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=config.actor_fc1, fc2_units=config.actor_fc2).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=config.actor_fc1, fc2_units=config.actor_fc2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Set up the Critic networks
        self.critic_local = Critic(state_size, action_size, seed, fc1_units=config.critic_fc1, fc2_units=config.critic_fc2).to(device)
        self.critic_target = Critic(state_size, action_size, seed, fc1_units=config.critic_fc1, fc2_units=config.critic_fc2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_decay)
    
        # Noise process for exploratary action
        self.noise = OUNoise((n_agents, action_size), seed)
    
        # Set replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, seed)
        
        # Copy over the weights
        self.hard_copy(self.actor_local, self.actor_target)
        self.hard_copy(self.critic_local, self.critic_target)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay buffer, and use a random batch from memory to learn."""
        # Add all experiences from all agents to memmory
        for i in range(self.n_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
        
        self.time_step += 1
        
        # If enough samples are availble in the buffer to sample, Learn
        if len(self.memory) > config.batch_size and self.time_step % config.update_every == 0:
            for i in range(config.num_update):
                experiences = self.memory.sample()
                self.learn(experiences, config.gamma)
            
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        action = np.zeros((self.n_agents, self.action_size))
        # Put model in evaluating mode
        self.actor_local.eval()
        #with torch.no_grad():
            #action = self.actor_local(state).cpu().data.numpy()
            
            
        with torch.no_grad():
            for i in range(self.n_agents):
                action_i = self.actor_local(state[i]).cpu().data.numpy()
                action[i, :] = action_i            
        # Put model back in training mode
        self.actor_local.train()
        # Add noise
        if add_noise:
            action += self.epsilon * self.noise.sample()
            #self.reset()
        return np.clip(action, -1, 1)
        
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        """Update policy and value using given batch of experiences given.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        Params:
        experiences: tuple of (s, a, r, s', done) tuples 
        gamma: discount factor        
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Update Critic
        # Get predicted next-state actions and Q values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimise the loss with gradient descent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Update Actor
        # Compute Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss with gradient descent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, config.tau)
        self.soft_update(self.actor_local, self.actor_target, config.tau)
        
        # Reset noise                               
        self.reset()
        # Decay epsilon
        self.epsilon -= config.epsilon_decay             
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params:
        local_model: model that weights will be copied from
        target_model: model that weights will be copied to
        tau: soft update parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def hard_copy(self, local_model, target_model):
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data)
            
    def save(self):
        torch.save(self.actor_local.state_dict(), 
                   str(config_actor_fc1)+'_'+str(config_actor_fc2) + '_actor.pth')
        torch.save(self.critic_local.state_dict(),
                   str(config.critic_fc1)+'_'+str(config.critic_fc2) + '_critic.pth')
    
    def load(self, actor_file, critic_file):
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.critic_local.load_state_dict(torch.load(critic_file))
        self.hard_copy(self.actor_local, self.actor_target)
        self.hard_copy(self.critic_local, self.critic_target)  
        

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
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the size of the internal memory."""
        return len(self.memory)