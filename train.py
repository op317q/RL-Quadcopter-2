## TODO: Train your agent here.
import sys
import pandas as pd
import numpy as np
from task2 import Task
from agents.agent import DDPG

num_episodes = 250
rewards = []

# Take off
init_pose=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_pose=np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])

# Hover
# init_pose=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
# target_pose=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

task = Task(init_pose=init_pose, 
            init_velocities=np.array([0.0, 0.0, 0.0]), 
            init_angle_velocities=np.array([0.0, 0.0, 0.0]),
            runtime=5., 
            target_pose=target_pose)
agent = DDPG(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.score, agent.best_score), end="")  # [debug]
            rewards.append(reward)
            break
    sys.stdout.flush()