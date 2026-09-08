import numpy as np
import Environment_Tree

#========================
# Tree Search Mouse BAMDP
#========================
# Parameters
ALPHA0 = 1.0          # Beta prior: α
BETA0  = 1.0          # Beta prior: β
N_GRID = 50           # number of grid points for θ
THETA_GRID = np.linspace(0, 1, N_GRID)
P_ESCAPE = 0.8        # probability of successful escape from P
R_SAFE = 1.0          # per-step reward at safe patch
R_DANGEROUS = 3.0     # per-step reward at dangerous patch
R_DEATH = -100.0      # terminal penalty on death
GAMMA = 0.95          # discount
HORIZON = 20          # planning horizon

# Belief prior
belief_theta = Environment_Tree.beta_prior_on_grid(THETA_GRID, ALPHA0, BETA0)
belief_theta = Environment_Tree.np_to_tuple(belief_theta)

# MDP
mdp = Environment_Tree.MouseMDP(
    states=[0, 1, 2, -1],
    state_to_action_space=[
        [0, 1],   # S
        [0, 1],   # D
        [0],      # P
        [],       # Death: terminal, no actions
    ],
    transition_fn=Environment_Tree.transitions_mouse,
    reward_fn=Environment_Tree.rewards_mouse,
    reward_last_fn=Environment_Tree.rewards_last,
    theta_grid=THETA_GRID,
    p_escape=P_ESCAPE,
    r_safe=R_SAFE,
    r_dangerous=R_DANGEROUS,
    r_death=R_DEATH,
    gamma=GAMMA,
)

# Initial belief state
h0 = Environment_Tree.BeliefState(s=0, belief_theta=belief_theta, t=0)

# Solve
list_of_h_sets = Environment_Tree.forward_pass(h0, HORIZON, mdp)
V, Q, pi, pi_theta = Environment_Tree.backward_pass(list_of_h_sets, mdp)   

#=========================================================================
# Test
if __name__ == "__main__":
    mdp = Environment_Tree.MouseMDP(
        states=[0, 1, 2, -1],
        state_to_action_space=[[0,1],[0,1],[0],[]],
        transition_fn=Environment_Tree.transitions_mouse,
        reward_fn=Environment_Tree.rewards_mouse,
        reward_last_fn=Environment_Tree.rewards_last,
        theta_grid=THETA_GRID,
        p_escape=0.8, r_safe=1.0, r_dangerous=3.0, r_death=-100.0, gamma=0.95,
    )
    h0 = Environment_Tree.BeliefState(s=0, belief_theta=Environment_Tree.np_to_tuple(
        Environment_Tree.beta_prior_on_grid(THETA_GRID, 1, 1)), t=0)

    sets = Environment_Tree.forward_pass(h0, 10, mdp)
    print(f"Nodes per layer: {[len(s) for s in sets]}")

    V, Q, pi, pi_theta = Environment_Tree.backward_pass(sets, mdp)
    print(f"V(h0) = {V[h0]:.4f}")
    print(f"π(h0) = {pi[h0]}")   