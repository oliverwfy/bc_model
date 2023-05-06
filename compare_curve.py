from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

directory_name = 'compare_ru_sprod/'
# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.2

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 1000

# simulation times
simulation_times = 100


# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1


# probability of receiving evidence
prob_evidence = 0.01


# percentage of malicious agents
malicious = 0.1
threshold= 0.5


noise = 0.1
# dampening
dampening = None

threshold_ls = [0.2, 0.5, 0.8]

for threshold in threshold_ls:

    pooling = True

    model = 'confidence_updating'
    malicious_type = 'fixed_belief'
    distance = 'total_variation'



    result_cu = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                            model=model, malicious_type=malicious_type, distance=distance,
                            k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                            malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                            dampening=dampening)

    model = 'sprod'

    result_sprod = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                  model=model, malicious_type=malicious_type, distance=distance,
                                  k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                  malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                  dampening=dampening)

    pooling = False
    result_evidence = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                     model=model, malicious_type=malicious_type, distance=distance,
                                     k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                     malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                     dampening=dampening)

    np.save(directory_name + f'avg_belief_evidence_only_threshold_{threshold}.npy', result_evidence['belief_avg_true_good'])
    np.save(directory_name + f'avg_belief_reliability_updating_threshold_{threshold}.npy', result_cu['belief_avg_true_good'])
    np.save(directory_name + f'avg_belief_sprod.npy_threshold_{threshold}', result_sprod['belief_avg_true_good'])

    plt.figure()
    plt.plot(range(max_iteration), result_evidence['belief_avg_true_good'].mean(axis=1),color='gray', linestyle='--')
    plt.plot(range(max_iteration), result_cu['belief_avg_true_good'].mean(axis=1))
    plt.plot(range(max_iteration), result_sprod['belief_avg_true_good'].mean(axis=1))

    plt.xlabel('time step', fontsize=14)
    plt.ylabel('average belief', fontsize=14)
    plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \beta = {prob_evidence}, \epsilon={noise}$', fontsize=14)
    plt.legend(['evidence only', 'reliability updating', 'equal weights'], fontsize=14)

    plt.savefig(directory_name + f'avg_belief_compare_threshold_{threshold}.png')