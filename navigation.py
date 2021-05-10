from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from nav_dqn_agent import Agent
import torch
import os
from model import QNetwork

# please do not modify the line below
root = os.path.dirname(__file__)
print(root)
path = root + "/Banana_Linux/Banana.x86_64"
print(path)
env = UnityEnvironment(file_name=path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# number of actions
action_size = brain.vector_action_space_size
# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)

# initialize the Nav Deep Q network agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

train = False
evaluate = True

def dqn(n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, train_m=True): #original 2000 episodes
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_m)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            done = env_info.local_done[0]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

if train:

    scores = dqn(n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, train_m=True)

if evaluate:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_model = QNetwork(state_size, action_size, 0).to(device)
    trained_model.load_state_dict(torch.load(root + "/checkpoint.pth"))
    trained_model.eval()
    
    agent.qnetwork_local = trained_model

    for i in range(10):

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:

            action = agent.act(state, 0.1)
            env_info = env.step(action)[brain_name]
            done = env_info.local_done[0]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break
        
        print("Evaluating Agent. Episode " + str(i) + " Score = " + str(score))
        

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()