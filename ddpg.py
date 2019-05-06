import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from configuration import Configuration

import torch
import torch.nn.functional as F
import torch.optim as optim


config = Configuration()


class DDPGAgent():
    """A class to create DDPG agents that interact and learn from the enviroment."""
    
    def __init__(self, state_size, action_size, index, seed):
        """Initilize the Agent.
        
        Params:
        state_size: dimension of the state
        action_size: dimension of the action
        seed: random seed
        """
        self.epsilon = config.epsilon
        self.index = index
        
        # Set up the Actor networks
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=config.actor_fc1, fc2_units=config.actor_fc2).to(config.device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=config.actor_fc1, fc2_units=config.actor_fc2).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Set up the Critic networks
        self.critic_local = Critic(state_size, action_size, seed, fc1_units=config.critic_fc1, fc2_units=config.critic_fc2).to(config.device)
        self.critic_target = Critic(state_size, action_size, seed, fc1_units=config.critic_fc1, fc2_units=config.critic_fc2).to(config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_decay)

        # Copy over the weights
        self.hard_copy(self.actor_local, self.actor_target)
        self.hard_copy(self.critic_local, self.critic_target)
    
        
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(config.device)
        # Put model in evaluating mode
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
                
        # Put model back in training mode
        self.actor_local.train()
        
        return action
        
        
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
                   str(config_actor_fc1)+'_'+str(config_actor_fc2) + str(self.index) + '_' + '_actor.pth')
        torch.save(self.critic_local.state_dict(),
                   str(config.critic_fc1)+'_'+str(config.critic_fc2) + str(self.index) + '_' + '_critic.pth')
    
    def load(self, actor_file, critic_file):
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.critic_local.load_state_dict(torch.load(critic_file))
        self.hard_copy(self.actor_local, self.actor_target)
        self.hard_copy(self.critic_local, self.critic_target)  
        