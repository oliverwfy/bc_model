from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
import pandas as pd


warnings.filterwarnings('ignore')

directory_name = 'consensus_heatmap/'
# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.2

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 5000

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
consensus_only = True


evidence_rate_ls = np.round(np.linspace(0.001, 0.01, 10), 3)
pool_size_ls = np.linspace(2, 20, 10).astype(int)

consensus_ru = []
consensus_eq = []
consensus_ru_mat = np.zeros((len(pool_size_ls), len(evidence_rate_ls)))
consensus_eq_mat = consensus_ru_mat.copy()

for i, k in enumerate(pool_size_ls):
    for j, prob_evidence in enumerate(evidence_rate_ls):
        pooling = True

        model = 'confidence_updating'
        malicious_type = 'fixed_belief'
        distance = 'kl'

        result_ru = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                model=model, malicious_type=malicious_type, distance=distance,
                                k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                dampening=dampening, consensus_only=consensus_only)

        consensus_ru_mat[i,j] = result_ru['consensus'].mean()

        model = 'sprod'

        result_eq = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                      model=model, malicious_type=malicious_type, distance=distance,
                                      k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                      malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                      dampening=dampening, consensus_only=consensus_only)

        consensus_eq_mat[i,j] = result_eq['consensus'].mean()


np.save(directory_name + f'consensus_eq_heatmap.npy', consensus_eq_mat)
np.save(directory_name + f'consensus_ru_heatmap.npy', consensus_ru_mat)


plt.figure()

sns.heatmap(pd.DataFrame(consensus_ru_mat, index=np.flip(pool_size_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)

# plt.xlabel(r'$\beta$', fontsize=14)
# plt.ylabel('k', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \epsilon={noise}$', fontsize=14)
# plt.legend(['equal weights', 'reliability updating'], fontsize=14)
plt.xticks(rotation=45)


plt.savefig(directory_name + f'consensus_time_heatmap_ru.png')

plt.figure()

sns.heatmap(pd.DataFrame(consensus_eq_mat, index=np.flip(pool_size_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)

# plt.xlabel(r'$\beta$', fontsize=14)
# plt.ylabel('k', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \epsilon={noise}$', fontsize=14)
# plt.legend(['equal weights', 'reliability updating'], fontsize=14)
plt.xticks(rotation=45)


plt.savefig(directory_name + f'consensus_time_heatmap_eq.png')