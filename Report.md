# Report
## Learning Algorithm
I used a MADDPG (Deep Deterministic Policy Gradient) learning algorithm for this project. For each agent, Two identical neural networks (identical in architecture) were used, one that does the learning and one set as target and soft update happens after a few time-steps of learning to make the target weights closer to the learning neural network. Same critic network was used for both agents.

### Hyper parameters
* Replay Buffer Size = 100000
* Batch Size = 512
* Gamma (Discount factor) = 0.99
* Tau (Soft update parameter) = 0.001
* Learning Rate for Actor Network = 0.001
* Learning Rate for Critic Network = 0.001
* Update Every = 2
* Number of Updates = 4
* number of episodes = 8000
* Epsilon Start = 1.0
* Epsilon Decay Rate =  0.000001
* Weight Decay = 0
* Noise Sigma = 2


### Model Architecture
Actor
`fc1 = [24, 256]`
`fc2 = [256, 256]`
`fc3 = [256, 4]`

Critic
`fc1 = [26, 256]`
`fc2 = [256, 256]`
`fc3 = [256, 1]`

## Plot of Rewards
Agent solved the environment in episode 5603! And achieved a maximum average score over last 100 episode of 0.92!

![](Assets/reward_plot.png?raw=true)

## Ideas for Future Work
* Make the agents learn from the raw pixels.
* Implement prioritized experience replay.
* Tune the hyper parameters further more.
