from a2c_test import test
from a2c_model import ActorCritic
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym

def train():
    gamma = 0.99
    lr = 0.0025
    betas = (0.99, 0.999)
    n_step = 3
    random_seed = 10
    
    torch.manual_seed(random_seed)
    
    env = gym.make('CartPole-v1')
    env.seed(random_seed)
    
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.n, hl_size=[128,128])
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    # running_reward = 0
    rewards_per_episode = [ [0]*500 for i in range(1000)]
    for i_episode in range(0, 1000):
        state = env.reset()
        running_reward = 0
        for t in range(500):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            rewards_per_episode[i_episode].append(reward)
            running_reward += reward
            if done:
                break
        
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(n=n_step, gamma=gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()
        
        # if running_reward > 4000:
        if env.unwrapped.spec.id == 'CartPole-v1':
            if running_reward > 495:
                print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
                torch.save(policy.state_dict(), './trained/model_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
                print("########## Solved! ##########")
                # break
        if env.unwrapped.spec.id == 'Acrobot-v1':
            if running_reward > -100:
                print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
                torch.save(policy.state_dict(), './trained/{}_params_{}_{}_{}_{}.pth'.format(env.unwrapped.spec.id, gamma, lr, betas[0], betas[1]))
                print("########## Solved! ##########")
                # break
        
        if i_episode % 20 == 0:
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))

    print(rewards_per_episode)
    variance_per_episode = [np.var(ep_rwd, ddof=1) for ep_rwd in rewards_per_episode]
    print(variance_per_episode)
    plt.plot(variance_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Variance of Episode Reward')
    plt.title('Variance of Episode Reward for each Episode')
    plt.show()

    test(env=env, policy=policy, save_video=True, name='{}_params_{}_{}_{}_{}'.format(env.unwrapped.spec.id, gamma, lr, betas[0], betas[1]))
            
if __name__ == '__main__':
    train()