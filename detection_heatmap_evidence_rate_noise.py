from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')

directory_name = 'detection_heatmap/'
# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.2

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 10000

# simulation times
simulation_times = 10


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
detection_only = True




pooling = True
malicious_type = 'fixed_belief'
distance = 'kl'


model = 'confidence_updating'

evidence_rate_ls = np.linspace(0, 0.02, 21)
pool_size_ls = np.linspace(2, 24, 23).astype(int)
malicious_ls = np.linspace(0, 0.2, 21)
noise_ls = np.linspace(0, 0.2, 21)


detection_ru_mat = np.zeros((len(noise_ls), len(evidence_rate_ls)))


for i, noise in enumerate(noise_ls):
    for j, prob_evidence in enumerate(evidence_rate_ls):
        pooling = True

        result_ru = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                model=model, malicious_type=malicious_type, distance=distance,
                                k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                dampening=dampening, consensus_only=consensus_only, detection_only=detection_only)

        detection_ru_mat[i,j] = result_ru['detection'].mean()


plt.figure()
np.save(directory_name + f'detection_heatmap_evidence_rate_noise.npy', detection_ru_mat)


sns.heatmap(pd.DataFrame(detection_ru_mat, index=np.flip(noise_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)

plt.savefig(directory_name + 'detection_heatmap_evidence_rate_noise.png')


# plt.figure()
# plt.plot(evidence_rate_ls, )
#
# plt.xlabel(r'time step', fontsize=14)
# plt.ylabel('acc', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \beta = {prob_evidence}, \epsilon={noise}$', fontsize=14)
# plt.legend(['equal weights', 'reliability updating'], fontsize=14)
#
# plt.savefig(directory_name + f'consensus_time_evidence_rate_0.3.png')