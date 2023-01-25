from model import simulate_model
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
mal_x = 0.2

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 5000

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

threshold = 0.5

noise = 0.2
# dampening
dampening = None

pool_ls = np.linspace(3, 15, 7).astype(int)


belief_evidence = []
belief_own = []
belief_pooled = []
belief_confidence = []
consensus_evidence = []
consensus_own = []
consensus_pooled = []
consensus_confidence = []


threshold_ls = [0.1, 0.3, 0.5, 0.7, 0.9]

k = 5

malicious = 0.1


for threshold in threshold_ls:
    pooling = False
    model = None
    malicious_type = None
    result_evidence = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                     model=model, malicious_type=malicious_type,
                                     k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                     malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                     dampening=dampening)
    belief_evidence.append(result_evidence['belief_avg_true_good'].mean(axis=1)[-1])
    consensus_evidence.append(int(result_evidence['consensus'].mean()))


    pooling = True
    model = 'bc_own_belief'
    malicious_type = 'fixed_belief'

    result_own = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                model=model, malicious_type=malicious_type,
                                k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                dampening=dampening)
    belief_own.append(result_own['belief_avg_true_good'].mean(axis=1)[-1])
    consensus_own.append(int(result_own['consensus'].mean()))

    pooling = True
    model = 'bc_pooled_belief'
    malicious_type = 'fixed_belief'
    result_pooled = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                   model=model, malicious_type=malicious_type,
                                   k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                   malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                   dampening=dampening)
    belief_pooled.append(result_pooled['belief_avg_true_good'].mean(axis=1)[-1])
    consensus_pooled.append(int(result_pooled['consensus'].mean()))

    pooling = True
    model = 'confidence_updating'
    malicious_type = 'fixed_belief'
    result_confidence = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                       model=model, malicious_type=malicious_type,
                                       k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                       malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                       dampening=dampening)
    belief_confidence.append(result_confidence['belief_avg_true_good'].mean(axis=1)[-1])
    consensus_confidence.append(int(result_confidence['consensus'].mean()))

plt.figure('avg belief (threshold)')

plt.plot(threshold_ls, belief_evidence, '--')
plt.plot(threshold_ls, belief_own)
plt.plot(threshold_ls, belief_pooled)
plt.plot(threshold_ls, belief_confidence)
plt.ylim(0, 1)
plt.legend(['evidence only', 'own belief', 'pooled belief', 'confidence updating'])
plt.title(f'average belief over {int(max_iteration)} iterations in different models')
plt.xlabel('pool size k')
plt.ylabel('avg belief')
plt.savefig(file_name + 'malicious_1_threshold_avg_belief.png')


plt.figure('consensus (threshold)')

plt.plot(threshold_ls, consensus_evidence, '--')
plt.plot(threshold_ls, consensus_own)
plt.plot(threshold_ls, consensus_pooled)
plt.plot(threshold_ls, consensus_confidence)
plt.ylim(0, max_iteration)
plt.legend(['evidence only', 'own belief', 'pooled belief', 'confidence updating'])
plt.title('consensus time in different models')
plt.xlabel('pool size k')
plt.ylabel('iteration')
plt.savefig(file_name + 'malicious_1_threshold_consensus.png')

