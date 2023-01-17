from model import simulate_malicious_acc, simulate_malicious_acc_bc
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
mal_x = 0.5

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 10

# simulation times
simulation_times = 1

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.01
weights_updating = 'kl_divergence'


# percentage of malicious agents
malicious = 0.2


file_name = 'compare_curve/'


threshold= 0.5

k = 10



mal_c = 0.5

noise = 0.1
# dampening
dampening = 0.001

strategy = 'deception'

malicious = 0.1


malicious_acc = []

pool_ls = [3, 5, 10,  15, 20]

mal_x_ls = [0, 0.02, 0.04, 0.06, 0.08, 0.1]

pooling = False

belief_evidence = []
belief = []
belief_bc = []


for malicious in mal_x_ls:

    result_evidence = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])

    pooling = True

    result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    belief.append(result['belief_avg_true_good'].mean(axis=1)[-1])


    result_bc = simulate_malicious_acc_bc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)



    belief_bc.append(result_bc['belief_avg_true_good'].mean(axis=1)[-1])




plt.figure(f'avg belief malicious')

plt.plot(mal_x_ls, belief_evidence, '--')
plt.plot(mal_x_ls, belief)
plt.plot(mal_x_ls, belief_bc)


plt.title(f'pool size = {k}, evidence rate = {prob_evidence}, noise = {noise}, threshold={threshold}')
plt.xlabel('percentage of malicious agents')
plt.ylabel('avg belief (after 1000 time steps)')


legend = ['evidence only', 'confidence updating', 'BC']
plt.legend(legend)
plt.show()

plt.savefig(file_name+f'compare_curve_malicious.png')



malicious = 0.05
pooling = False


belief_evidence = []
belief = []
belief_bc = []


for k in pool_ls:

    result_evidence = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])

    pooling = True

    result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    belief.append(result['belief_avg_true_good'].mean(axis=1)[-1])


    result_bc = simulate_malicious_acc_bc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)



    belief_bc.append(result_bc['belief_avg_true_good'].mean(axis=1)[-1])




plt.figure(f'avg belief pool size')

plt.plot(pool_ls, belief_evidence, '--')
plt.plot(pool_ls, belief)
plt.plot(pool_ls, belief_bc)


plt.title(f'malicious = {int(malicious*100)}%, evidence rate = {prob_evidence}, noise = {noise}, threshold={threshold}')
plt.xlabel('pool size k')
plt.ylabel('avg belief (after 1000 time steps)')


legend = ['evidence only', 'confidence updating', 'BC']
plt.legend(legend)
plt.show()

plt.savefig(file_name+f'compare_curve_k.png')