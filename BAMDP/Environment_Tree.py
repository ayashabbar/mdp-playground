import numpy as np
from dataclasses import dataclass, field
from scipy.stats import beta
#import Environment
from typing import Any, Callable

@dataclass(frozen=True)
class BeliefState:
    s: int
    belief_theta: tuple[float, ...]
    t: int

@dataclass
class MouseMDP:
    states: list
    state_to_action_space: list
    transition_fn: Callable[..., Any]
    reward_fn: Callable[..., Any]
    reward_last_fn: Callable[..., Any]
    theta_grid: np.array
    p_escape: float
    r_safe: float
    r_dangerous: float
    r_death: float
    gamma: float   

def expand_node_action(h, a, h_set, mdp):
    t = h.t
    transition_fn = mdp.transition_fn
    theta_grid = mdp.theta_grid
    p_escape = mdp.p_escape
   
    next_s, next_belief_theta, _ = transition_fn(h, a, theta_grid, p_escape)

    belief_states = [
        BeliefState(s=s, belief_theta=belief_theta, t=t+1)
        for s, belief_theta in zip(next_s, next_belief_theta)
    ]
    h_set.update(belief_states)


def forward_pass(h0, horizon, mdp):
    belief_state_sets = []
    belief_state_sets.append(set([h0]))

    for d in range(horizon):

        set_depth_d = set()

        for h in belief_state_sets[d]:
            action_space = mdp.state_to_action_space[h.s]
            for a in action_space:
                expand_node_action(h, a, set_depth_d, mdp)

        belief_state_sets.append(set_depth_d)

    return belief_state_sets

def solve_node(h, V, Q, pi, pi_theta, mdp):
    # Memoization
    if h in V:
       return V[h]

    # Terminal state (death): assign terminal value, skip Bellman
    if h.s == -1:
        V[h] = mdp.reward_last_fn(h.s, mdp.r_death)
        return V[h]

    action_space = mdp.state_to_action_space[h.s]
    transition_fn = mdp.transition_fn
    theta_grid = mdp.theta_grid
    gamma = mdp.gamma
    r_safe = mdp.r_safe
    r_dangerous = mdp.r_dangerous
    r_death = mdp.r_death
    p_escape = mdp.p_escape

    Q_h = []
    t = h.t
    expected_theta_t = (t, h.s, np.sum(h.belief_theta * theta_grid))

    for a in action_space:

        # Transition: returns next observable states, updated beliefs, and probabilities
        next_s, next_belief_theta, prob_next_s = transition_fn(h, a, theta_grid, p_escape)

        # Immediate reward: biting cheese at current state
        if h.s == 0:          # Safe patch
            r = r_safe
        elif h.s == 1:        # Dangerous patch
            r = r_dangerous
        else:                 # Predator present (fleeing, no cheese)
            r = 0.0

        # Build successor belief states
        belief_states = [
            BeliefState(s=s, belief_theta=belief_theta, t=t+1)
            for s, belief_theta in zip(next_s, next_belief_theta)
        ]

        # Bellman backup
        next_V = np.array([V[next_h] for next_h in belief_states])
        Q_h_a = np.sum(prob_next_s * (r + gamma * next_V))
        Q_h.append(Q_h_a)

    Q_h = np.array(Q_h)
    a_max = np.argmax(Q_h)
    Q_max = np.max(Q_h)

    Q[h] = Q_h
    V[h] = Q_max
    pi[h] = a_max
    pi_theta[expected_theta_t] = a_max   


def backward_pass(list_of_h_sets, mdp):
    V = dict()
    Q = dict()
    pi = dict()
    pi_theta = dict()

    # Initialize terminal values
    r_last = mdp.reward_last_fn
    for h in list_of_h_sets[-1]:
        V[h] = r_last(h.s)

    for hs in list_of_h_sets[-2::-1]:
        for h in hs:
            solve_node(h, V, Q, pi, pi_theta, mdp)

    return V, Q, pi, pi_theta   

def np_to_tuple(x):
    return tuple(x.tolist())

def rewards_mouse(s, a, theta):
    import Environment
    if s == 0: return Environment.R_SAFE
    if s == 1: return Environment.R_DANGEROUS
    if s == 2: return 0.0   # no reward while escaping

def rewards_last(state, r_death=-100.0):
    import Environment
    if state == -1:
        return Environment.R_DEATH
    return 0.0     

def transitions_mouse(h, a, theta_grid, p_escape):
    s = h.s
    belief = np.asarray(h.belief_theta)
    next_s = []
    next_belief = []
    prob_next_s = []

    if s == 0:  # Safe
        if a == 1:
            next_s.append(1)
        else:
            next_s.append(0)
        next_belief.append(np_to_tuple(belief))
        prob_next_s.append(1.0)

    elif s == 1:  # Dangerous
        if a == 0:  # stay → Bernoulli(θ)
            b1 = belief * (1 - theta_grid); b1 /= b1.sum()
            b2 = belief * theta_grid;       b2 /= b2.sum()
            # Probability of each outcome is the belief-weighted expectation:
            p_predator = float(np.sum(belief * theta_grid))
            next_s.extend([1, 2])
            next_belief.extend([np_to_tuple(b1), np_to_tuple(b2)])
            prob_next_s.extend([1.0 - p_predator, p_predator])
        else:  # retreat
            next_s.append(0)
            next_belief.append(np_to_tuple(belief))
            prob_next_s.append(1.0)

    elif s == 2:  # Predator present → escape
        next_s.extend([0, -1])
        next_belief.extend([np_to_tuple(belief)] * 2)
        prob_next_s.extend([p_escape, 1.0 - p_escape])

    return next_s, next_belief, prob_next_s     

def beta_prior_on_grid(theta_grid, a, b):
    theta_grid = np.asarray(theta_grid)
    N = len(theta_grid)
    dw = np.zeros(N)  # compute bin widths
    dw[1:-1] = 0.5 * (theta_grid[2:] - theta_grid[:-2])
    dw[0] = 0.5 * (theta_grid[1] - theta_grid[0])
    dw[-1] = 0.5 * (theta_grid[-1] - theta_grid[-2])
    # convert to probability mass
    pdf_vals = beta.pdf(theta_grid, a, b)
    p = pdf_vals * dw
    p /= p.sum()
    return p



