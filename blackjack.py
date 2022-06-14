import gym
from collections import defaultdict

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


# define first-visit mc prediction function
def fv_mc_prediction(policy, env, n_episodes):
    # init the empty value table - dictionary to store values of each state
    value_table = defaultdict(float)
    N = defaultdict(int)

    # for each episode generate data and store
    for _ in range(n_episodes):
        # generate & store data
        states, _, rewards = generate_episode_data(policy, env)
        returns = 0;

        # for each step store rewards and states to temp variable and calculate returns as sum of rewards
        # for each episode calculate current value of all the states involved (starting from terminal state)

        for i in range(len(states) -1, -1, -1):
            _reward = rewards[i]
            _state = states[i]

            # add reward to returns
            returns += _reward

            # for first-visit check if episode is visited for the first time
            if _state not in states[:i]:
                # if yes standard mc incremental equation
                # NewEstimate = OldEstimate + StepSize(Target-OldEstimate)
                N[_state] += 1
                value_table[_state] += (returns - value_table[_state]) / N[_state]

    return value_table








