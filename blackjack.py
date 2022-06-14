import gym


# init gym environment
env = gym.make('Blackjack-v0')


# define policy to hit until 19 (0=stand, 1=hit)
def hit_til_19(observation):
    score, dealer_score, ace = observation
    return 0 if score >= 19 else 1
