from model import simulate_bc_own_belief_malicious_2, simulate_bc_pooled_belief_malicious_2
import matplotlib.pyplot as plt
import warnings

import numpy as np
warnings.filterwarnings('ignore')

font_size = 16

# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = None

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 1000

# simulation times
simulation_times = 50


# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1


# probability of receiving evidence
prob_evidence = 0.01


# percentage of malicious agents
malicious = 0.1


file_name = 'malicious_one/'


threshold= 0.5


noise = 0.2
# dampening
dampening = None


pool_ls = np.linspace(3,15,7).astype(int)


belief_evidence = []
belief_own = []
belief_pooled = []
for k in pool_ls:
    pooling = False

    result_evidence = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                         k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                         malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])



    pooling = True
    result_own = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                    k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                    malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_own.append(result_own['belief_avg_true_good'].mean(axis=1)[-1])


    pooling = True
    result_pooled = simulate_bc_pooled_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                          k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                          malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_pooled.append(result_pooled['belief_avg_true_good'].mean(axis=1)[-1])

plt.figure('pool size')

plt.plot(pool_ls, belief_evidence, '--')
plt.plot(pool_ls, belief_own)
plt.plot(pool_ls, belief_pooled)
plt.ylim(0,1)
plt.legend(['evidence only', 'own belief', 'pooled belief'])
plt.title('BC model with different distance')
plt.xlabel('pool size k')
plt.ylabel('avg belief')
plt.savefig(file_name + 'bc_malicious_2_pool_size.png')







k = 10

mal_x_ls = [0, 0.02, 0.04, 0.06, 0.08, 0.1]



belief_evidence = []
belief_own = []
belief_pooled = []
for malicious in mal_x_ls:
    pooling = False
    result_evidence = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                         k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                         malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])



    pooling = True
    result_own = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                    k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                    malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_own.append(result_own['belief_avg_true_good'].mean(axis=1)[-1])


    pooling = True
    result_pooled = simulate_bc_pooled_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                          k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                          malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_pooled.append(result_pooled['belief_avg_true_good'].mean(axis=1)[-1])

plt.figure('malicious')
plt.plot(mal_x_ls, belief_evidence, '--')
plt.plot(mal_x_ls, belief_own)
plt.plot(mal_x_ls, belief_pooled)

plt.ylim(0,1)

plt.legend(['evidence only', 'own belief', 'pooled belief'])
plt.title('BC model with different distance')
plt.xlabel('percentage of malicious agents')
plt.ylabel('avg belief')
plt.savefig(file_name + 'bc_malicious_2_malicious.png')










threshold_ls = [0.1,0.3, 0.5, 0.7, 0.9]

k = 10

malicious = 0.1

belief_evidence = []
belief_own = []
belief_pooled = []
for threshold in threshold_ls:
    pooling = False
    result_evidence = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                         k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                         malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])



    pooling = True
    result_own = simulate_bc_own_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                    k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                    malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_own.append(result_own['belief_avg_true_good'].mean(axis=1)[-1])


    pooling = True
    result_pooled = simulate_bc_pooled_belief_malicious_2(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                                          k=k, init_x = init_x, mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                                          malicious = malicious, threshold= threshold, noise=noise, pooling=pooling, dampening = dampening)
    belief_pooled.append(result_pooled['belief_avg_true_good'].mean(axis=1)[-1])

plt.figure('threshold')
plt.plot(threshold_ls, belief_evidence, '--')
plt.plot(threshold_ls, belief_own)
plt.plot(threshold_ls, belief_pooled)

plt.ylim(0,1)

plt.legend(['evidence only', 'own belief', 'pooled belief'])
plt.title('BC model with different distance')
plt.xlabel('threshold')
plt.ylabel('avg belief')
plt.savefig(file_name + 'bc_malicious_2_threshold.png')

