import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    '''
    Shared Actor-Critic Network.
    '''
    def __init__(self, input_dim, output_dim, hl_size = [1024, 512]):
        super(ActorCritic, self).__init__()

        dim = input_dim
        self.hidden_layers = nn.ModuleList()
        
        # Code for Dynamically adding layers.
        for hl_dim in hl_size:
            self.hidden_layers.append(nn.Linear(dim, hl_dim))
            dim = hl_dim
        
        self.action_layer = nn.Linear(dim, output_dim) # Second param Action_space
        self.value_layer = nn.Linear(dim, 1)
        
        self.log_probs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        '''
        "forward()" function is pretty straight forward.
        We are using Categorical sampling as the Action Distribution for sampling actions.
        '''
        state = torch.from_numpy(state).float()

        for layer in self.hidden_layers:
            state = F.relu(layer(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.log_probs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, n = 1, gamma = 0.99):
        '''
        This function calculates the loss for our Actor-Critic Network.
        The default values of Gamma as given in the Question have been kept.
        Args:
            n:
                -1 : Full Return
                >0 : n-step return
        '''
        # Full return || Case with (n = -1)
        if n == -1:
            rewards = []
            
            # Discounting the Rewards:
            dis_reward = 0
            for reward in self.rewards[::-1]:
                dis_reward = reward + gamma * dis_reward
                rewards.insert(0, dis_reward)
            
            # Normalizing the rewards:
            rewards = torch.tensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std())
            
            # Implementing the Algorithm for Full Returns (Monte Carlo)
            loss = 0
            for log_prob, value, reward in zip(self.log_probs, self.state_values, rewards):
                d_t = reward  - value.item()
                action_loss = -log_prob * d_t
                value_loss = (d_t ** 2)
                loss += (action_loss + value_loss)
                
            return loss
        
        else:
            loss = 0        
            
            for t, state_value in enumerate(zip(self.log_probs, self.state_values)):
                log_prob, value = state_value
                d_t = 0
                
                # n-Step TD Return
                # Summation t'=t to t'=n+t-1
                for tp in range(t, min(n+t, len(self.rewards)-1)):
                    d_t += ((gamma**(tp - t)) * self.rewards[tp]) / n
                
                if min(n+t, len(self.rewards)-1) == n+t:
                    d_t += ((gamma**n) * self.state_values[n+t])
                
                d_t = d_t - value
                
                action_loss = -log_prob * d_t
                value_loss = (d_t ** 2)
                loss += (action_loss + value_loss)  
            
            return loss 

    def clearMemory(self):
        '''
        Function to clear the log_probs, state_values, and rewards.
        '''
        del self.log_probs[:]
        del self.state_values[:]
        del self.rewards[:]