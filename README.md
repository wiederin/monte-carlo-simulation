# Monte Carlo Simulation

Monte Carlo analysis or simulation involves running many scenarios with different random inputs and summarizing the
distribution of the results. The larger the sample of random inputs the more accurate the predicted outcome. Monte Carlo
simulations can be used to estimate the expected return when starting in state s, taking action a, and following policy
pi - q(s, a, pi). Monte Carlo methods are classified as 'First-visit' or 'Every visit'. The difference between the two
being the number of times a state can be visited within an episode before an MC update is made. The first-visit MC
method estimates the value of all states as the average of the returns following first visits to each state before
termination, whereas the every-visit MC method averages the returns following an n-number of visits to a state before
termination.

## Monte Carlo procedure:

```
Input: a policy P to be evaluates

Initialize:
    V(s) ∈ R, arbitrarily, for all s ∈ S
    Returns(s) <- an empty list, for all s ∈ S

Loop forever (for each episode):
    Generate an episode following P: S0, A0, R1, S1, A1, R2, ... , ST-1, AT-1, RT
    G <- 0
    Loop for each step to episode, t = T - 1, T - 2, ... , 0:
        G <-  γG + Rt+1
        Unless St appears in S0, S1, ..., St-1:
            Append G to Returns(St)
            V(St) <- average(Returns(St))
```


## Incremental Monte Carlo update procedure

<img src="https://latex.codecogs.com/svg.image?\color{white}V(S_{t})&space;\underset{}{\leftarrow}&space;V(S_{t})&space;&plus;&space;\alpha&space;[G(S_{t})&space;-&space;V(S_{t})]">


## First-Visit MC method implementation in Blackjack or 21

* uses OpenAI's gym environment (interface for running games of blackjack - all collected information on states, actions
, and rewards are kept within "observation" variables)

## Sources

* https://towardsdatascience.com/optimizing-blackjack-strategy-through-monte-carlo-methods-cbb606e52d1b

