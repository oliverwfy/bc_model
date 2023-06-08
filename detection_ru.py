from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

directory_name = 'detection/'
# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.2

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 2000

# simulation times
simulation_times = 100


# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1


# probability of receiving revidence
prob_evidence = 0.01


# percentage of malicious agents
malicious = 0.1
threshold= 0.5


noise = 0.1
# dampening
dampening = None

# stop iteration while reach consensus
consensus_only = False

# stop iteration while detect all malicious agents
detection_only = False




pooling = True
malicious_type = 'fixed_belief'
distance = 'kl'


model = 'confidence_updating'


result_ru = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                        model=model, malicious_type=malicious_type, distance=distance,
                        k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                        malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                        dampening=dampening, consensus_only=consensus_only, detection_only=detection_only)



acc = result_ru['accuracy']
precision = result_ru['precision']
recall = result_ru['recall']

acc_individual = result_ru['accuracy_individual']
precision_individual = result_ru['precision_individual']
recall_individual = result_ru['recall_individual']

np.save(directory_name + f'acc_swarm_evidence_rate_{prob_evidence}.npy', acc)
np.save(directory_name + f'precision_swarm_evidence_rate_{prob_evidence}.npy', precision)
np.save(directory_name + f'recall_swarm_evidence_rate_{prob_evidence}.npy', recall)
np.save(directory_name + f'acc_individual_evidence_rate_{prob_evidence}.npy', acc_individual)
np.save(directory_name + f'precision_individual_evidence_rate_{prob_evidence}.npy', precision_individual)
np.save(directory_name + f'recall_individual_evidence_rate_{prob_evidence}.npy', recall_individual)




# plt.figure()
# plt.plot(evidence_rate_ls, )
#
# plt.xlabel(r'time step', fontsize=14)
# plt.ylabel('acc', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \beta = {prob_evidence}, \epsilon={noise}$', fontsize=14)
# plt.legend(['equal weights', 'reliability updating'], fontsize=14)
#
# plt.savefig(directory_name + f'consensus_time_evidence_rate_0.3.png')