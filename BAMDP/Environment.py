import os

import numpy as np
import matplotlib.pyplot as plt

# Allows Environment.py to be imported from outside the BAMDP folder AND run directly
try:
    from . import Environment_Tree
except ImportError:
    import Environment_Tree


def simulate(pi, mdp, h0, theta_true, horizon=20, n_episodes=100, seed=None):
    """
    Roll out the learned policy under a fixed ground-truth θ.
    """
    if seed is not None:
        np.random.seed(seed)

    all_returns = []
    all_trajectories = []

    for ep in range(n_episodes):
        s = h0.s
        belief = np.asarray(h0.belief_theta)
        t = 0
        total_return = 0.0
        trajectory = [(s, belief.copy(), t)]

        while s != -1 and t < horizon:
            h = Environment_Tree.BeliefState(
                s=s, belief_theta=Environment_Tree.np_to_tuple(belief), t=t
            )
            if h in pi:
                a = pi[h]
            else:
                a = np.random.choice(mdp.state_to_action_space[s])

            if s == 0:
                s_prime = 1 if a == 1 else 0
                r = mdp.r_safe
            elif s == 1:
                if a == 0:  # stay / bite -> Bernoulli(θ_true)
                    if np.random.rand() < theta_true:
                        s_prime = 2
                        belief = belief * mdp.theta_grid
                        belief /= belief.sum()
                    else:
                        s_prime = 1
                        belief = belief * (1.0 - mdp.theta_grid)
                        belief /= belief.sum()
                else:  # retreat
                    s_prime = 0
                r = mdp.r_dangerous
            elif s == 2:
                if np.random.rand() < mdp.p_escape:
                    s_prime = 0
                else:
                    s_prime = -1
                r = 0.0

            total_return += mdp.gamma**t * r
            t += 1
            s = s_prime
            trajectory.append((s, belief.copy(), t))

            if s == -1:
                total_return += mdp.gamma**t * mdp.r_death
                break

        all_returns.append(total_return)
        all_trajectories.append(trajectory)

    return all_returns, all_trajectories


def figure_bamdp(show=False, out_dir="outputs"):
    if not show:
        os.makedirs(out_dir, exist_ok=True)
    # Parameters
    ALPHA0 = 1.0
    BETA0 = 9.0
    N_GRID = 50
    THETA_GRID = np.linspace(0, 1, N_GRID)
    P_ESCAPE = 0.8
    R_SAFE = 1.0
    R_DANGEROUS = 6.0
    R_DEATH = -25.0
    GAMMA = 0.95
    HORIZON = 20

    # Belief prior & MDP setup
    belief_theta = Environment_Tree.beta_prior_on_grid(THETA_GRID, ALPHA0, BETA0)
    belief_theta = Environment_Tree.np_to_tuple(belief_theta)

    action_spaces = [
        list((0, 1)),  # S
        list((0, 1)),  # D
        list((0,)),  # P
        list(),  # Death
    ]

    mdp = Environment_Tree.MouseMDP(
        states=[0, 1, 2, -1],
        state_to_action_space=action_spaces,
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

    # Solve BAMDP
    h0 = Environment_Tree.BeliefState(s=0, belief_theta=belief_theta, t=0)
    list_of_h_sets = Environment_Tree.forward_pass(h0, HORIZON, mdp)
    V, Q, pi, pi_theta = Environment_Tree.backward_pass(list_of_h_sets, mdp)

    # Simulate at fixed theta_true = 0.4
    theta_true = 0.4
    returns, trajs = simulate(pi, mdp, h0, theta_true, horizon=HORIZON, n_episodes=500)
    print(f"Mean discounted return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean episode length:    {np.mean([len(t) - 1 for t in trajs]):.1f}")

    # --- Figure 1: Return distribution, trajectory, and belief evolution ---
    fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax0, ax1, ax2 = axes

    ax0.hist(returns, bins=30, edgecolor="black", alpha=0.7)
    ax0.axvline(
        np.mean(returns), color="red", linestyle="--", label=f"μ={np.mean(returns):.1f}"
    )
    ax0.set_xlabel("Discounted return")
    ax0.set_ylabel("Count")
    ax0.set_title(f"Returns (θ={theta_true:.2f})")
    ax0.legend()

    traj = trajs[0]
    states = [step[0] for step in traj]
    times = [step[2] for step in traj]
    state_labels = {0: "S", 1: "D", 2: "P", -1: "Death"}
    y = [state_labels[s] for s in states]

    ax1.plot(times, y, "o-", color="steelblue")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(["S", "D", "P", "Death"])
    ax1.set_xlabel("t")
    ax1.set_title("Example trajectory")

    beliefs = [np.asarray(step[1]) for step in traj]
    mean_theta = [np.sum(b * THETA_GRID) for b in beliefs]

    ci_lo, ci_hi = [], []
    for b in beliefs:
        cdf = np.cumsum(b)
        ci_lo.append(THETA_GRID[np.searchsorted(cdf, 0.05)])
        ci_hi.append(THETA_GRID[np.searchsorted(cdf, 0.95)])

    ax2.fill_between(times, ci_lo, ci_hi, alpha=0.3, color="orange", label="90% CrI")
    ax2.plot(times, mean_theta, "k-", label="E[θ]")
    ax2.axhline(
        theta_true, color="red", linestyle=":", label=f"true θ={theta_true:.2f}"
    )
    ax2.set_xlabel("t")
    ax2.set_ylabel("θ")
    ax2.set_title("Belief over θ")
    ax2.legend()

    plt.tight_layout()

    # --- Figure 2: Sweep θ and plot mean return ---
    theta_sweep = np.linspace(0.01, 0.99, 50)
    mean_returns = []
    for th in theta_sweep:
        rets, _ = simulate(pi, mdp, h0, th, horizon=HORIZON, n_episodes=200)
        mean_returns.append(np.mean(rets))

    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(theta_sweep, mean_returns, "b-", linewidth=2)
    plt.xlabel("True θ")
    plt.ylabel("Mean discounted return")
    plt.title("Performance vs. true θ")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        fig1.savefig("simulation_2.png", dpi=150)
        fig2.savefig("BAMDP_2.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    figure_bamdp(show=True)
