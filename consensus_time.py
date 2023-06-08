from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

directory_name = 'consensus/'
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
consensus_only = True


evidence_rate_ls = np.linspace(0.001, 0.03, 30)


consensus_ru = []
consensus_eq = []
consensus_bc = []

pooling = True
malicious_type = 'fixed_belief'
distance = 'kl'

for prob_evidence in evidence_rate_ls:


    model = 'confidence_updating'


    result_ru = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                            model=model, malicious_type=malicious_type, distance=distance,
                            k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                            malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                            dampening=dampening, consensus_only=consensus_only)

    consensus_ru.append(result_ru['consensus'].mean())

    model = 'sprod'

    result_eq = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                  model=model, malicious_type=malicious_type, distance=distance,
                                  k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                  malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                  dampening=dampening, consensus_only=consensus_only)

    consensus_eq.append(result_eq['consensus'].mean())

    # model = 'bc'
    #
    # result_bc = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
    #                               model=model, malicious_type=malicious_type, distance=distance,
    #                               k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
    #                               malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
    #                               dampening=dampening, consensus_only=consensus_only)
    #
    # consensus_bc.append(result_bc['consensus'].mean())



plt.figure()
plt.plot(evidence_rate_ls, consensus_eq)
plt.plot(evidence_rate_ls, consensus_ru)

plt.xlabel(r'evidence rate ($\beta$)', fontsize=14)
plt.ylabel('consensus time', fontsize=14)
plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \beta = {prob_evidence}, \epsilon={noise}$', fontsize=14)
plt.legend(['equal weights', 'reliability updating'], fontsize=14)
np.save(directory_name + f'consensus_eq_evidence_rate_0.3.npy', consensus_eq)
np.save(directory_name + f'consensus_ru_evidence_rate_0.3.npy', consensus_ru)

plt.savefig(directory_name + f'consensus_time_evidence_rate_0.3.png')