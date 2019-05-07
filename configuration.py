import torch


# Replay buffer size
BUFFER_SIZE = int(1e5)
# Minibatch size
BATCH_SIZE = 512
# Discount Factor
GAMMA = 0.99
# Soft update parameter
TAU = 1e-3
# Actor learning rate
LR_ACTOR = 1e-3
# Critic learning rate
LR_CRITIC = 1e-3
# L2 weight decay
WEIGHT_DECAY = 0
# Number of time steps before each update
UPDATE_EVERY = 2
# Number of updates in each update
NUM_UPDATE = 4
# Epsilon for the noise process added to the actions
EPSILON = 1
# Epsilon decay rate
EPSILON_DECAY = 1e-6
# Number of nodes in Actor network
ACTOR_FC1 = 512
ACTOR_FC2 = 512
# Number of nodes in Critic network
CRITIC_FC1 = 512
CRITIC_FC2 = 512


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Configuration():
    """A class to save the hyperparameters and configs."""
    
    def __init__(self):
        """Initialize the class."""
        # Buffer Params
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        
        # Discount factors
        self.gamma = GAMMA
        
        # Soft update param
        self.tau = TAU
        
        # Training Params
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.weight_decay = WEIGHT_DECAY
        self.update_every = UPDATE_EVERY
        self.num_update = NUM_UPDATE
        
        # Epsilon params
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        
        # Network params
        self.actor_fc1 = ACTOR_FC1
        self.actor_fc2 = ACTOR_FC2
        self.critic_fc1 = CRITIC_FC1
        self.critic_fc2 = CRITIC_FC2
        
        # Set training device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")