import gym
import wandb

def test(env, policy, n_episodes=5, render=False, save_video = False, name=''):
    env = env
    policy = policy

    if save_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{name}")
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
        wandb.log({"eval_reward": running_reward})
    env.close()
            
if __name__ == '__main__':
    test()