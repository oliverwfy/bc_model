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
consensus_only = True

detection_only = False



evidence_rate_ls = np.round(np.linspace(0, 0.02, 21), 4)
pool_size_ls = np.linspace(2, 24, 23).astype(int)
malicious_ls = np.round(np.linspace(0, 0.2, 21),2)
noise_ls = np.round(np.linspace(0, 0.2, 21),3)


consensus_ru_mat = np.zeros((len(noise_ls), len(evidence_rate_ls)))
consensus_eq_mat = consensus_ru_mat.copy()
consensus_bc_mat = consensus_ru_mat.copy()

malicious_type = 'fixed_belief'
distance = 'kl'
pooling = True

for i, noise in enumerate(noise_ls):
    for j, prob_evidence in enumerate(evidence_rate_ls):

        # model = 'confidence_updating'
        # result_ru = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
        #                         model=model, malicious_type=malicious_type, distance=distance,
        #                         k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
        #                         malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
        #                         dampening=dampening, consensus_only=consensus_only, detection_only=detection_only)
        #
        # consensus_ru_mat[i,j] = result_ru['consensus'].mean()

        # model = 'sprod'
        #
        # result_eq = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
        #                               model=model, malicious_type=malicious_type, distance=distance,
        #                               k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
        #                               malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
        #                               dampening=dampening, consensus_only=consensus_only,detection_only=detection_only)
        #
        # consensus_eq_mat[i,j] = result_eq['consensus'].mean()
        #
        model = 'bc_own_belief'


        result_bc = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                      model=model, malicious_type=malicious_type, distance=distance,
                                      k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                      malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                      dampening=dampening, consensus_only=consensus_only, detection_only=detection_only)

        consensus_bc_mat[i,j] = result_bc['consensus'].mean()


# np.save(directory_name + f'consensus_ru_heatmap_evidence_rate_noise.npy', consensus_ru_mat)
# np.save(directory_name + f'consensus_eq_heatmap_evidence_rate_noise.npy', consensus_eq_mat)
np.save(directory_name + f'consensus_bc_heatmap_evidence_rate_noise.npy', consensus_bc_mat)


# plt.figure()
#
# sns.heatmap(pd.DataFrame(consensus_ru_mat, index=np.flip(noise_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)
#
# # plt.xlabel(r'$\beta$', fontsize=14)
# # plt.ylabel('k', fontsize=14)
# # plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \epsilon={noise}$', fontsize=14)
# # plt.legend(['equal weights', 'reliability updating'], fontsize=14)
# plt.xticks(rotation=45)
#
#
# plt.savefig(directory_name + f'consensus_time_heatmap_ru_evidence_rate_noise.png')
#


# plt.figure()
#
# sns.heatmap(pd.DataFrame(consensus_eq_mat, index=np.flip(noise_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)
#
# # plt.xlabel(r'$\beta$', fontsize=14)
# # plt.ylabel('k', fontsize=14)
# # plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \epsilon={noise}$', fontsize=14)
# # plt.legend(['equal weights', 'reliability updating'], fontsize=14)
# plt.xticks(rotation=45)
#
#
# plt.savefig(directory_name + f'consensus_time_heatmap_eq_evidence_rate_noise.png')


plt.figure()

sns.heatmap(pd.DataFrame(consensus_bc_mat, index=np.flip(noise_ls), columns=evidence_rate_ls), vmin=0, vmax=max_iteration)

# plt.xlabel(r'$\beta$', fontsize=14)
# plt.ylabel('k', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \epsilon={noise}$', fontsize=14)
# plt.legend(['equal weights', 'reliability updating'], fontsize=14)
plt.xticks(rotation=45)


plt.savefig(directory_name + f'consensus_time_heatmap_bc_evidence_rate_noise.png')

