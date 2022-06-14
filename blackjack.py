import gym


# init gym environment
env = gym.make('Blackjack-v0')


# define policy to hit until 19 (0=stand, 1=hit)
def hit_til_19(observation):
    score, dealer_score, ace = observation
    return 0 if score >= 19 else 1


# define method to generate data for an episode
def generate_episode_data(policy, env):
    # init the lists to store states, actions, and rewards
    states, actions, rewards = [], [], []

    # reset gym environment
    observation = env.reset()

    while True:
        # add the states to resp. list
        states.append(observation)
        # select action based on policy
        action = hit_til_19(observation)
        # append to actions list
        actions.append(action)
        # perform action in environment
        observation, reward, done, info = env.step(action)
        # add the reward to rewards list
        rewards.append(reward)
        # break if state is terminal
        if done:
            break
        return states, actions, rewards




