import torch
import torch.optim as optim
import gym
import wandb
import numpy as np

from a2c_test import test
from a2c_model import ActorCritic

def train(CONFIG):
    gamma = float(CONFIG.gamma)
    lr = float(CONFIG.lr)
    beta1, beta2 = float(CONFIG.beta1), float(CONFIG.beta2)
    betas = (beta1, beta2)
    seed = int(CONFIG.seed)
    n_step = int(CONFIG.n_step)
    num_episodes = 1000
    hl_size = CONFIG.hl_size
    
    torch.manual_seed(seed)
    
    env = gym.make(str(CONFIG.env_id))
    env.seed(seed)
    
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.n, hl_size=hl_size)

    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    # running_reward = 0
    rewards_per_episode = [ [0]*500 for i in range(1000)]
    for i_episode in range(0, num_episodes):
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

        wandb.log({"episode_length": t,"episodic_reward": running_reward})
        
        if env.unwrapped.spec.id == 'CartPole-v1':
            if running_reward > 495:
                print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
                print("########## Solved! ##########")
        if env.unwrapped.spec.id == 'Acrobot-v1':
            if running_reward > -100:
                print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
                print("########## Solved! ##########")
        
        if i_episode % 20 == 0:
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))

    for ep_rwd in rewards_per_episode:
        wandb.log({"var_ep_rwd": np.var(ep_rwd, ddof=1)})
    test(env=env, policy=policy, save_video=True, name='{}_params_{}_{}_{}_{}'.format(env.unwrapped.spec.id, gamma, lr, betas[0], betas[1]))

def sweep():
    config_defaults = dict(
            # env_id = "CartPole-v1",
            env_id = "Acrobot-v1",
            exp_name = "a2c",
            seed = 11,
            n_step = -1,
            gamma = 0.99,
            lr = 0.00025,
            hl_size=[20,20],
            beta1 = 0.99,
            beta2 = 0.999
        )
        
    wandb.init(config = config_defaults)

    run_name = str(wandb.config.env_id) + "_" + str(wandb.config.exp_name) + "_" + str(wandb.config.seed) + "_td_" + str(wandb.config.n_step) + "_hl_" + str(wandb.config.hl_size) + "_lr_" + str(wandb.config.lr) + "_beta1_" + str(wandb.config.beta1) + "_beta2_" + str(wandb.config.beta2)

    wandb.run.name = run_name

    CONFIG = wandb.config
    print(CONFIG)

    train(CONFIG)
            
if __name__ == '__main__':
    wandb.login()
    
    sweep_config = {
        # "name": "CartPole-v1 Grid Sweep",
        "name": "Acrobot-v1 Grid Sweep",
        "method": "grid",
        "metric":
        {
            "name": "episodic_reward",
            "goal": "maximize"
        },
        "parameters":
        {
            # "env_id" : {"values": ["CartPole-v1"]},
            "env_id" : {"values": ["Acrobot-v1"]},
            # "env_id" : {"values": ["MountainCar-v0"]},
            "seed" : {"values": [11]}, # Fixed
            "n_step": {"values": [-1,1,3,5,20,25,35]},
            # "lr" : {"values": [0.00025, 0.001, 0.0025, 0.003]}, # CartPole-v1
            "lr" : {"values": [0.0005, 0.0001, 0.001, 0.002, 0.003, 0.05]}, # Acrobot-v1
            # "lr" : {"values": [0.0005, 0.0001, 0.0003, 0.001, 0.05, 0.003]}, # MountainCar-v0
            # "hl_size": {"values": [[20,15], [15,15], [20,10], [20,25], [32,32], [128]]}, # CartPole-v1
            "hl_size": {"values": [[64,64], [32,64,64], [128,64], [512,256], [1024, 512]]}, # Acrobot-v1
            # "hl_size": {"values": [[32,8],[32,32]]}, # MountainCar-v0
            "gamma" : {"values": [0.99]}, # Fixed
            "beta1": {"values": [0.99]}, # Fixed
            "beta2": {"values": [0.999]}, # Fixed
            "save_model": {"values": [False]} # Fixed
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="RL_PA-2", entity="argha-jash")
    wandb.agent(sweep_id, function=sweep)
    wandb.finish()

