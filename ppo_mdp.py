from helper import *
from protagonist import Protagonist

import math
import random

import gymnasium as gym

import numpy as np
import time 
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt

from collections import deque
from protagonist_mdp import Protagonist
from parallel_env import make_env
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

num_envs = 30
device = torch.device(0 if torch.cuda.is_available() else "cpu")
print("Using device:", device)
lr = 0.0001
# How many minibatchs (therefore optimization steps) we want per epoch 
num_mini_batch = 8
# Total number of steps during the rollout phase 
num_steps = 256 
# Number of Epochs for training
ppo_epochs = 4

# PPO parameters
gamma = 0.995
tau = 0.95
clip_param = 0.2



def ppo_update(data_buffer, ppo_epochs, clip_param):
    for _ in range(ppo_epochs):
        for data_batch in data_buffer:
            # Forward pass using the method without hidden state management
            # During training, we don't maintain LSTM hidden states across timesteps
            new_dist, new_value = rl_model(data_batch["states"], data_batch["directions"])

            # Most Policy gradient algorithms include a small "Entropy bonus" to increases the "entropy" of 
            # the action distribution, aka the "randomness"
            # This ensures that the actor does not converge to taking the same action everytime and
            # maintains some ability for "exploration" of the policy
            
            # Determine expectation over the batch of the action distribution entropy
            entropy = new_dist.entropy().mean()

            actor_loss = ppo_loss(new_dist, data_batch["actions"], data_batch["log_probs"], data_batch["advantages"],
                                  clip_param)

            critic_loss = clipped_critic_loss(new_value, data_batch["values"], data_batch["returns"], clip_param)

            # These techniques allow us to do multiple epochs of our data without huge update steps throwing off our
            # policy/value function (gradient explosion etc).
            # It can also help prevent "over-fitting" to a single batch of observations etc, 
            # RL boot-straps itself and the noisy "ground truth" targets (if you can call them that) will
            # shift overtime and we need to make sure our actor-critic can quickly adapt, over-fitting to a
            # single batch of observations will prevent that
            agent_loss = critic_loss - actor_loss

            optimizer.zero_grad()
            agent_loss.backward()
            # Clip gradient norm to further prevent large updates
            nn.utils.clip_grad_norm_(rl_model.parameters(), 40)
            optimizer.step()


# Training parameters
max_frames = 3e7
frames_seen = 0
rollouts = 0

# Score loggers
test_score_logger = []
train_score_logger = []
frames_logger = []

# Set the size of the FIFO databuffer to the total number of steps for a single batch of rollouts
# so that every episode the buffer has completely reset
buffer_size = num_steps * num_envs
# Calculate the size of each minibatch  - usually very big - 2048!
mini_batch_size = buffer_size // num_mini_batch

# Define the data we wish to collect for the databuffer
data_names = ["states", "actions", "log_probs", "values", "returns", "advantages", "directions"]
data_buffer = ReplayBuffer(data_names, buffer_size, mini_batch_size, device)

# Create the actor critic Model and optimizer
rl_model = Protagonist().to(device)
optimizer = optim.Adam(rl_model.parameters(), lr=lr)

envs = SyncVectorEnv([make_env() for _ in range(num_envs)])
start_time = time.time()

while frames_seen < max_frames:
    rl_model.train()
    # Initialise state
    start_state, _ = envs.reset()  #[30,7,7,3]
    obs = start_state['image']
    direction = torch.tensor(start_state['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
    state = state_to_tensor(obs, device)

    # Create data loggers - deques a bit faster than lists
    log_probs = deque()
    values = deque()
    states = deque()
    actions = deque()
    rewards = deque()
    masks = deque()
    directions = deque()

    step = 0
    cnt = 0
    done = np.zeros(num_envs)
    print("Rollout!")
    with torch.no_grad():  # Don't need computational graph for roll-outs
        while step < num_steps:
            #  Masks so we can separate out multiple games in the same environment
            dist, value = rl_model(state, direction)  # Forward pass of actor-critic model
            action = dist.sample()  # Sample action from distribution

            # Take the next step in the env
            next_state, reward, termination, truncation, info = envs.step(action.cpu().numpy())
            done = np.logical_or(termination, truncation)

            cnt += (reward>0).sum()
            # Reset hidden states for environments that finished
            # done is a numpy array of shape (num_envs,) with True/False values
            
            # Log data
            reward = torch.tensor(reward, dtype=torch.float32).to(device)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            states.append(state)  # [num_steps,num_envs,3,7,7]
            actions.append(action)
            values.append(value)
            rewards.append(reward.unsqueeze(1).to(device))
            current_mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            masks.append(current_mask)
            directions.append(direction)

            direction = torch.tensor(next_state['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
            next_state = next_state['image']
            state = state_to_tensor(next_state, device)
            step += 1

        # Get value at time step T+1
        _, next_value = rl_model(state, direction)
        # Calculate the returns/gae
        returns, advantage = compute_gae(next_value, rewards, masks, values, gamma=gamma, tau=tau)

        data_buffer.data_log("states", torch.cat(list(states)))
        data_buffer.data_log("actions", torch.cat(list(actions)))
        data_buffer.data_log("returns", torch.cat(list(returns)))
        data_buffer.data_log("log_probs", torch.cat(list(log_probs)))
        data_buffer.data_log("values", torch.cat(list(values)))
        data_buffer.data_log("directions", torch.cat(list(directions)))
        advantage = torch.cat(list(advantage)).squeeze(1)
        # Normalising the Advantage helps stabalise training!
        data_buffer.data_log("advantages", (advantage - advantage.mean()) / (advantage.std() + 1e-8))
        
        # Update the frames counter
        # We normaly base how long to train for by counting the number of "environment interactions"
        # In our case we can simply counte how many game frames we have received from the environment
        frames_seen += advantage.shape[0]
    
    # We train after every batch of rollouts
    # With the stabalisation techniques in PPO we can "safely" take many steps with a single
    # batch of rollouts, therefore we usualy train with the data over multiple epochs whereas basic
    # actor critic methods only use one epoch.
    print("Training")
    ppo_update(data_buffer, ppo_epochs, clip_param)
    rollouts += 1
    print(cnt)
    if rollouts % 1 == 0:
        # print("Testing")
        rl_model.eval()
        # TODO: Implement testing for custom environment
        # test_score = evaluate_agent(env, rl_model, device)
        # train_score = run_tests(train_test="train")
        # test_score = 0  # Placeholder
        # train_score = 0  # Placeholder

        # test_score_logger.append(test_score)
        # frames_logger.append(frames_seen)
        # print("Trained on %d Frames, Test Score [%d/%d]" 
        #     %(frames_seen, test_score))
        if rollouts%290 == 0:
            torch.save(rl_model.state_dict(), f"LSTM_Protagonist_{rollouts}.pth")
        print("Trained on %d Frames" 
            %(frames_seen))
        time_to_end = ((time.time() - start_time) / frames_seen) * (max_frames - frames_seen)
        print("Time to end: %dh:%dm" % (time_to_end // 3600, (time_to_end % 3600) / 60))



torch.save(rl_model.state_dict(), f"LSTM_Protagonist.pth")
print(f" training finsihed for seed. saved!")

print("Done!")

#1. protagonist ko architecture change because, now we have 30 env
#2. Ani ho reset vayepaxi, hidden k hunxa? (jun env jaile ni reset hunu sakxa, hidden ta zero chiyo initially)