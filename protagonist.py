# Architecture of Protagonist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class Protagonist(nn.Module):
    def __init__(self, num_actions=3):
        super(Protagonist, self).__init__()


        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) # (5,5,16) = 400
        self.dir_fc = nn.Linear(4, 5)

        # Lstm input = 400 + 5 = 405
        self.lstm = nn.LSTM(input_size=405, hidden_size=256, batch_first=True)

        # 2 fully connected layers
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)

        # Policy 
        self.policy = nn.Linear(32, num_actions)

        # Value
        self.value = nn.Linear(32, 1)

    def forward(self, obs, direction, hidden=None):
        """
        obs: tensor (batch, 3, 7, 7)
        direction: tensor (batch,) with values 0..3
        hidden: LSTM hidden state (h,c), optional
        """
        batch_size = obs.size(0)
        x = obs
        x = F.relu(self.conv(x))      # (batch, 16, 5, 5)
        x = x.reshape(batch_size, -1)    # (batch, 400)

        # One-hot encode direction
        dir_onehot = F.one_hot(direction, num_classes=4).float()
        d = F.relu(self.dir_fc(dir_onehot))  # (batch, 5)

        # Concat
        features = torch.cat([x, d], dim=-1)  # (batch, 405)

        # Add time dimension for LSTM (batch, seq=1, feat)
        features = features.unsqueeze(1)

        # LSTM
        lstm_out, hidden = self.lstm(features, hidden)  # (batch, 1, 256)
        h = lstm_out.squeeze(1)  # (batch, 256)

        # FC layers
        h = F.relu(self.fc1(h))  # (batch, 32)
        h = F.relu(self.fc2(h))  # (batch, 32)

        # Outputs
        policy_logits = self.policy(h)   # (batch, 3)
        value = self.value(h)            # (batch, 1)

        dist = Categorical(logits = policy_logits)
        return dist, value, hidden

    def forward_no_lstm(self, obs, direction):
        """
        Forward pass without LSTM hidden state management.
        Used during training when we don't need to maintain hidden states.
        
        obs: tensor (batch, 3, 7, 7)
        direction: tensor (batch,) with values 0..3
        """ 
        batch_size = obs.size(0)
        x = obs
        x = F.relu(self.conv(x))      # (batch, 16, 5, 5)
        x = x.reshape(batch_size, -1)    # (batch, 400)

        # One-hot encode direction
        dir_onehot = F.one_hot(direction, num_classes=4).float()
        d = F.relu(self.dir_fc(dir_onehot))  # (batch, 5)

        # Concat
        features = torch.cat([x, d], dim=-1)  # (batch, 405)

        # Add time dimension for LSTM (batch, seq=1, feat)
        features = features.unsqueeze(1)

        # Use zero hidden state for training
        h0 = torch.zeros(1, batch_size, 256, device=obs.device)
        c0 = torch.zeros(1, batch_size, 256, device=obs.device)
        
        # LSTM
        lstm_out, _ = self.lstm(features, (h0, c0))  # (batch, 1, 256)
        h = lstm_out.squeeze(1)  # (batch, 256)

        # FC layers
        h = F.relu(self.fc1(h))  # (batch, 32)
        h = F.relu(self.fc2(h))  # (batch, 32)

        # Outputs
        policy_logits = self.policy(h)   # (batch, 3)
        value = self.value(h)            # (batch, 1)

        dist = Categorical(logits = policy_logits)
        return dist, value


# TODO:
# what is the best initial hidden parameter for lstm?
# simulate the time steps here aafaile
# see how will this be trained. 

# agent = Protagonist(3)
# initial_hidden = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
# dummy_obs = torch.randn(1, 7, 7, 3)  # (batch, 7, 7, 3)
# dummy_dir = torch.tensor([0])        # (batch,)
# policy_logits, value, new_hidden = agent(dummy_obs, dummy_dir, initial_hidden)
# print("Policy logits:", policy_logits.detach())

# policy_logits_1, value_1, new_hidden_1 = agent(dummy_obs, dummy_dir, new_hidden)
# print("policy_logits_1:", policy_logits_1.detach())
# print("Value:", value_1.detach())


# observations haruko ani action haruko info data buffer ma save garne  ho
# so yo save garne bela ma lstm chai halyo
# aba policy update ko bela ma ni lstm ta sure update hunxa
# kasari hunxa?
