import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os

def main():
    environment_name = "CarRacing-v2"
    env = gym.make(environment_name)

    # episodes = 5
    # for episode in range(1, episodes+1):
    #     state = env.reset()
    #     done = False
    #     score = 0 
        
    #     while not done:
    #         env.render()
    #         action = env.action_space.sample()
    #         n_state, reward, done, info, _ = env.step(action)
    #         score+=reward
    #     print('Episode:{} Score:{}'.format(episode, score))
    # env.close()

    log_path = os.path.join('Training', 'Logs')
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=40000)


    evaluate_policy(model, env, n_eval_episodes=10, render=True)
    env.close()

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info, _ = env.step(action)
        env.render()
        
    env.close()
        
if __name__ == "__main__":
    main()