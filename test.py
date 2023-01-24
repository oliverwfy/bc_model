from model import *


# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.2

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 1

# simulation times
simulation_times = 1


# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1


# probability of receiving evidence
prob_evidence = 0.01


# percentage of malicious agents
malicious = 0.1
threshold= 0.5


noise = 0.2
# dampening
dampening = None

pooling = True

result_confidence = simulate_confidence_malicious_1(simulation_times=simulation_times, pop_n=pop_n,
                                                    max_iteration=max_iteration, k=k, init_x=init_x,
                                                    classification=True,
                                                    dampening=dampening, alpha=alpha, prob_evidence=prob_evidence,
                                                    malicious=malicious, mal_x=mal_x, threshold=threshold, noise=noise,
                                                    pooling=pooling)


