import gym
from collections import defaultdict

# init gym environment
env = gym.make('Blackjack-v1')


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
        # print(reward)
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
        data = generate_episode_data(policy, env)
        returns = 0
        if data is not None:
            # for each step store rewards and states to temp variable and calculate returns as sum of rewards
            # for each episode calculate current value of all the states involved (starting from terminal state)
            for i in range(len(data[0]) - 1, -1, -1):
                _reward = data[2][i]
                _state = data[0][i]

                # add reward to returns
                returns += _reward
                # for first-visit check if episode is visited for the first time
                if _state not in data[0][:i]:
                    # if yes standard mc incremental equation
                    # NewEstimate = OldEstimate + StepSize(Target-OldEstimate)
                    N[_state] += 1
                    value_table[_state] += (returns - value_table[_state]) / N[_state]
            # print(returns)
    return value_table


prediction = fv_mc_prediction(hit_til_19, env, n_episodes=500000)


for j in range(10):
    print(prediction.popitem())

