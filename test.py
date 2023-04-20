from model import simulate_model
import matplotlib.pyplot as plt


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
simulation_times = 5


# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1


# probability of receiving evidence
prob_evidence = 0.01


# percentage of malicious agents
malicious = 0.1
threshold= 0.7


noise = 0.1
# dampening
dampening = None

pooling = True

model = 'confidence_updating'
malicious_type = 'fixed_belief'
distance = 'kl'


result_bc_own_belief_tv = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                 model=model, malicious_type=malicious_type, distance=distance,
                                 k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                 malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                 dampening=dampening)

distance = 'kl'

result_bc_own_belief_kl = simulate_model(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration,
                                         model=model, malicious_type=malicious_type, distance=distance,
                                         k=k, init_x=init_x, mal_x=mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                         malicious=malicious, threshold=threshold, noise=noise, pooling=pooling,
                                         dampening=dampening)


plt.plot(range(max_iteration), result_bc_own_belief_kl['accuracy'].mean(axis=1))
plt.plot(range(max_iteration), result_bc_own_belief_kl['precision'].mean(axis=1))
plt.plot(range(max_iteration), result_bc_own_belief_kl['recall'].mean(axis=1))


plt.legend(['acc' ,'pre', 'recall'])
plt.show()