from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from DDPG_Agent import DDPG_Agent


# get the default brain
env = UnityEnvironment(file_name='./Reacher20.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

import matplotlib.pyplot as plt
env_info = env.reset(train_mode=True)[brain_name]
agent = DDPG_Agent(state_size=state_size, action_size=action_size, seed=8)


if __name__ == "__main__":
    scores = []  # initialize the score

    train_mode = False
    #agent = DDPG_Agent(state_size=env_info.vector_observations.shape[1], action_size=brain.vector_action_space_size, random_seed=8)

    if train_mode:
        for i_eposide in range(2000):
            env_info = env.reset(train_mode=True)[brain_name]
            num_agents = len(env_info.agents)
            states = env_info.vector_observations
            episode_scores = np.zeros(num_agents)
            time_step = 1
            while True:
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]
                next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done

                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                episode_scores += rewards
                time_step +=1
                if np.any(dones):
                    break

            episode_mean_score = np.mean(episode_scores)
            scores.append(episode_mean_score)
            print('\rEpisode {}\t Avg Score: {:.2f}\t Max Score: {:.2f}\t Min Score: {:.2f}'.format(i_eposide, episode_mean_score, np.max(episode_scores), np.min(episode_scores)))

            if len(scores) > 100:
                print('\rEpisode {}\t Score: {:.2f} \t Avg Score: {:.2f}'.format(i_eposide, episode_mean_score, np.mean(scores[-100:])))

            if len(scores)>100 and np.mean(scores[-100:]) >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score over last 100 Episodes: {:.2f}'.format(i_eposide, np.mean(scores[-100:])))
                torch.save(agent.actor_net_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_net_local.state_dict(), 'checkpoint_critic.pth')
                plt.figure()
                plt.plot(range(i_eposide+1), scores)
                plt.xlabel("Episode")
                plt.ylabel("Score")
                plt.show()

                break
    else:
        agent.actor_net_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        agent.critic_net_local.load_state_dict(torch.load('checkpoint_critic.pth'))
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        episode_scores = np.zeros(num_agents)  # initialize the score
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done # see if episode has finished
            episode_scores += rewards  # update the score
            states = next_states  # roll over the state to next time step
            if np.any(dones):  # exit loop if episode finished
                print("Score: {:.2f}".format(np.mean(episode_scores)))
                break
