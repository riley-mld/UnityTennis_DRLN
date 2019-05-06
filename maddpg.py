import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from agent import DDPGAgent

import torch
import torch.nn.functional as F
import torch.optim as optim


class MADDPGAgent():
    """A class to create MADDPG agents, to train in the enviroment and compete/collabrate."""
    def __init__(self)