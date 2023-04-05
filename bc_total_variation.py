from model import simulate_bc_own_belief_malicious_1_total_variation
import numpy as np


alpha = 0.1
init_x = None
mal_x = 0.3
malicious = 0.1
threshold = 0.2

result = simulate_bc_own_belief_malicious_1_total_variation(simulation_times=1, pop_n=10, max_iteration=50, k=3, init_x = init_x,
                                                            mal_x = mal_x,alpha=alpha, prob_evidence=0.05, malicious=malicious, threshold= threshold,
                                                            noise=None, pooling=True, dampening = False)


confidence_mat = result['confidence_mat']
np.save('confidence_mat.npy', confidence_mat)
