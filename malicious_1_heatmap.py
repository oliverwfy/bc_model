from model import simulate_model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
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
simulation_times = 100

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.01

# percentage of malicious agents
malicious = 0.1

file_name = 'heatmap/'

threshold = 0.5

noise = 0.2
# dampening
dampening = None

k = 5

mal_x_ls = np.round(np.linspace(0.05,0.5,10),2)
threshold_ls = np.round(np.linspace(0.1,0.9,9), 2)


belief_evidence = []
belief_own = []
belief_pooled = []
belief_confidence = []
consensus_evidence = []
consensus_own = []
consensus_pooled = []
consensus_confidence = []


belief_mat_evidence = np.zeros((len(mal_x_ls), len(threshold_ls)))
consensus_mat_evidence = np.copy(belief_mat_evidence)
belief_mat_own = np.zeros((len(mal_x_ls), len(threshold_ls)))
consensus_mat_own = np.copy(belief_mat_evidence)
belief_mat_pooled = np.zeros((len(mal_x_ls), len(threshold_ls)))
consensus_mat_pooled = np.copy(belief_mat_evidence)
belief_mat_confidence = np.zeros((len(mal_x_ls), len(threshold_ls)))
consensus_mat_confidence = np.copy(belief_mat_evidence)




for i, mal_x in enumerate(np.flip(mal_x_ls)):
    for j, threshold in enumerate(threshold_ls):

        pooling = False
        model = None
        malicious_type = None
        result_evidence = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                         model=model, malicious_type=malicious_type,
                                         k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                         malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                         dampening=dampening)
        belief_mat_evidence[i,j] = result_evidence['belief_avg_true_good'].mean(axis=1)[-1]
        consensus_mat_evidence[i,j] = int(result_evidence['consensus'].mean())

        #
        # pooling = True
        # model = 'bc_own_belief'
        # malicious_type = 'fixed_belief'
        #
        # result_own = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
        #                             model=model, malicious_type=malicious_type,
        #                             k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
        #                             malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
        #                             dampening=dampening)
        # belief_mat_own[i,j] = result_own['belief_avg_true_good'].mean(axis=1)[-1]
        # consensus_mat_own[i,j] = int(result_own['consensus'].mean())
        #
        # pooling = True
        # model = 'bc_pooled_belief'
        # malicious_type = 'fixed_belief'
        # result_pooled = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
        #                                model=model, malicious_type=malicious_type,
        #                                k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
        #                                malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
        #                                dampening=dampening)
        # belief_mat_pooled[i,j] = result_pooled['belief_avg_true_good'].mean(axis=1)[-1]
        # consensus_mat_pooled[i,j] = int(result_pooled['consensus'].mean())
        #
        # pooling = True
        # model = 'confidence_updating'
        # malicious_type = 'fixed_belief'
        # result_confidence = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
        #                                    model=model, malicious_type=malicious_type,
        #                                    k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
        #                                    malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
        #                                    dampening=dampening)
        # belief_mat_confidence[i,j] = result_confidence['belief_avg_true_good'].mean(axis=1)[-1]
        # consensus_mat_confidence[i,j] = int(result_confidence['consensus'].mean())



np.save(file_name + 'npy/heatmap_avg_belief_evidence.npy', belief_mat_evidence)
np.save(file_name + 'npy/heatmap_consensus_time_evidence.npy', consensus_mat_evidence)

np.save(file_name + 'npy/heatmap_avg_belief_own.npy', belief_mat_own)
np.save(file_name + 'npy/heatmap_consensus_time_own.npy', consensus_mat_own)


np.save(file_name + 'npy/heatmap_avg_belief_pooled.npy', belief_mat_pooled)
np.save(file_name + 'npy/heatmap_consensus_time_pooled.npy', consensus_mat_pooled)


np.save(file_name + 'npy/heatmap_avg_belief_confidence.npy', belief_mat_confidence)
np.save(file_name + 'npy/heatmap_consensus_time_confidence.npy', consensus_mat_confidence)


plt.figure('avg belief evidence')

sns.heatmap(pd.DataFrame(belief_mat_evidence, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=1)
plt.title(f"average belief over {int(max_iteration)} iterations")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_avg_belief_evidence.png')


plt.figure('consensus evidence')
sns.heatmap(pd.DataFrame(consensus_mat_evidence, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=max_iteration)
plt.title(f"time to reach consensus")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_consensus_time_evidence.png')



plt.figure('avg belief own')

sns.heatmap(pd.DataFrame(belief_mat_own, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=1)
plt.title(f"average belief over {int(max_iteration)} iterations")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_avg_belief_own.png')


plt.figure('consensus own')
sns.heatmap(pd.DataFrame(consensus_mat_own, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=max_iteration)
plt.title(f"time to reach consensus")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_consensus_time_own.png')


plt.figure('avg belief pooled')

sns.heatmap(pd.DataFrame(belief_mat_pooled, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=1)
plt.title(f"average belief over {int(max_iteration)} iterations")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_avg_belief_pooled.png')


plt.figure('consensus pooled')
sns.heatmap(pd.DataFrame(consensus_mat_pooled, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=max_iteration)
plt.title(f"time to reach consensus")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_consensus_time_pooled.png')


plt.figure('avg belief confidence')

sns.heatmap(pd.DataFrame(belief_mat_confidence, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=1)
plt.title(f"average belief over {int(max_iteration)} iterations")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_avg_belief_confidence.png')


plt.figure('consensus confidence')
sns.heatmap(pd.DataFrame(consensus_mat_confidence, index=np.flip(mal_x_ls), columns=threshold_ls), vmin=0, vmax=max_iteration)
plt.title(f"time to reach consensus")
plt.xlabel('threshold')
plt.ylabel('mal_x')
plt.savefig(file_name + 'heatmap_consensus_time_confidence.png')

