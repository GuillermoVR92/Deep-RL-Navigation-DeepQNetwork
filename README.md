## Navigation Banana Environment description

Guillermo del Valle Reboul

## Project details

The goal of the agent is to collect as much `yellow` bananas as possible while avoiding the `blue` ones.

Unity brain name: BananaBrain
Number of Visual Observations (per agent): 0
Vector Observation space type: continuous
Vector Observation space size (per agent): 37
Number of stacked Vector Observation: 1
Vector Action space type: discrete
Vector Action space size (per agent): 4
Vector Action descriptions: , , , 


state vector = 37 continous values
actions = 4 discrete actions (forward, backward, turn left, turn right)
The environment is considered solved when agents reaches average score of 13.0 on 100 consecutive episodes.

## Algorithm Used: DQN

A Deep Q Network DQN was used for this project. The DQN model is built with 4 Linear layers fully connected (nn.Linear size 256) and Relu activation functions.
Hyperparameters:
* Replay buffer size = 100000
* Batch size = 256
* GAMMA (discount factor) = 0.998
* Tau = 0.001
* Learning rate = 0.0005
* Update the network evey 16 step

</br>

## Plot of Rewards

DQN Solved the environment in less than 600 episodes (Average score = 13.0). See episode_training.JPG

## Instructions

For executing Navigation.ipynb, "model.py" and "nav_dqn_agent.py" must be located in the same path as well as the Unity Banana executable file located in
file_name="/data/Banana_Linux_NoVis/Banana.x86_64"

For executing navigation.py (training and evaluation), locate all files (model.py , nav_dqn_agent.py , navigation.py) in same folder and also add the path "/Banana_Linux/Banana.x86_64"
so that the environment can be loaded.

Select flag "train" and "evaluate" to True / False depending on the desired execution.
