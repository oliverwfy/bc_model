from utility import *
from sklearn.metrics import accuracy_score, precision_score, recall_score


def simulate_model(simulation_times=100, pop_n=100, max_iteration=10000, k=3, init_x = 0.5, model = 'bc_own_belief', malicious_type = 'fixed_belief',
                           mal_x = 0.5,alpha=0.5, prob_evidence=0.02, malicious=0.0, threshold= 0.5,
                           noise=None, pooling=True, dampening = False):

    consensus = np.empty(simulation_times)
    belief_avg_true_good = np.empty([max_iteration, simulation_times])
    accuracy_swarm = belief_avg_true_good.copy()


    print('--------------Simulation Starts--------------')
    print('Simulation times : {}'.format(simulation_times))
    print('Malicious : {}'.format(malicious))
    if mal_x:
        print(f'Malicious Belief: {mal_x}\n')
    else:
        print(f'Malicious Belief: Uni(0,1)\n')


    if not pooling:
        print('Evidence Only:')
        print('Params: alpha = {} \n'.format(alpha))

    else:
        print('Log-linear operator:')
        print('Params: k = {} alpha = {}\n'.format(k, alpha))

    for n in range(simulation_times):
        consensus_time = None
        pop = np.array([Agent(pop_n, _, init_x,  True) for _ in range(pop_n)])
        true_mal_id = generate_malfunctioning_agents(pop, malicious, init_x=mal_x)

        true_mal_ls = np.zeros(pop_n)
        true_mal_ls[true_mal_id] = 1

        for i in range(max_iteration):

            # average belief of normal agents
            belief_avg_true_good[i,n] = avg_belief_good(true_mal_id, pop)
            if not consensus_time and belief_avg_true_good[i,n] >= 0.99:
                consensus_time = i
            # evidential updating
            for agent in pop:
                if np.random.random() <= prob_evidence and agent.state:
                    if noise and np.random.random() <= noise:
                        evidential_update(agent, 1-alpha, dampening)
                    else:
                        evidential_update(agent, alpha, dampening)



            if pooling:

                # pool selection
                pool = np.random.choice(pop, size=k, replace=False)
                # opinion pooling
                if malicious_type == 'fixed_belief':
                    if model == 'bc_own_belief':
                        opinion_pooling_own_belief_malicious_1(pool, threshold=threshold)
                    elif model == 'bc_pooled_belief':
                        opinion_pooling_pooled_belief_malicious_1(pool, threshold=threshold)
                    elif model == 'confidence_updating':
                        opinion_pooling_confidence_malicious_1(pool, threshold=threshold)

                elif malicious_type == 'min_rule':
                    if model == 'bc_own_belief':
                        opinion_pooling_own_belief_malicious_2(pool, threshold=threshold)
                    elif model == 'bc_pooled_belief':
                        opinion_pooling_pooled_belief_malicious_2(pool, threshold=threshold)
                    elif model == 'confidence_updating':
                        opinion_pooling_confidence_malicious_2(pool, threshold=threshold)


                if model == 'confidence_updating':
                    pred_mal_ls = np.zeros(pop_n)

                    for agent in pop:
                        pred_mal_ls += agent.mal_detection()

                    # an agent is malfunctioning if it is labeled by half of agents
                    pred_mal_id = np.where(pred_mal_ls >= pop_n/2)[0]
                    pred_mal_ls = np.zeros(pop_n)
                    pred_mal_ls[pred_mal_id] = 1

                    # swarm's prediction
                    accuracy_swarm[i,n] = accuracy_score(true_mal_ls, pred_mal_ls)

        consensus[n] = int(consensus_time) if consensus_time else max_iteration

    print('----------------Simulation ends----------------\n\n')

    result = {'belief_avg_true_good': belief_avg_true_good, 'consensus' : consensus, 'accuracy' : accuracy_swarm}
    return result




def simulate_bc_own_belief_malicious_1_total_variation(simulation_times=100, pop_n=10, max_iteration=1000, k=3, init_x = 0.5,
                                                       mal_x = 0.5,alpha=0.5, prob_evidence=0.02, malicious=0.0, threshold= 0.5,
                                                       noise=None, pooling=True, dampening = False):

    belief_avg_true_good = np.empty([max_iteration, simulation_times])
    confidence_mat = np.ones((simulation_times, max_iteration, pop_n, pop_n))

    print('--------------Simulation Starts--------------')
    print('Simulation times : {}'.format(simulation_times))
    print('Malicious : {}'.format(malicious))
    if mal_x:
        print(f'Malicious Belief: {mal_x}\n')
    else:
        print(f'Malicious Belief: Uni(0,1)\n')


    if not pooling:
        print('Evidence Only:')
        print('Params: alpha = {} \n'.format(alpha))

    else:
        print('Log-linear operator:')
        print('Params: k = {} alpha = {}\n'.format(k, alpha))

    for n in range(simulation_times):
        pop = np.array([Agent(pop_n, _, init_x,  True) for _ in range(pop_n)])
        true_mal_id = generate_malfunctioning_agents(pop, malicious, init_x=mal_x)

        true_mal_ls = np.zeros(pop_n)
        true_mal_ls[true_mal_id] = 1

        for i in range(max_iteration):

            # average belief of normal agents
            belief_avg_true_good[i,n] = avg_belief_good(true_mal_id, pop)

            # evidential updating
            for agent in pop:
                if np.random.random() <= prob_evidence and agent.state:
                    if noise and np.random.random() <= noise:
                        evidential_update(agent, 1-alpha, dampening)
                    else:
                        evidential_update(agent, alpha, dampening)

            # opinion pooling
            opinion_pooling_own_belief_malicious_1_total_variation(pop, confidence_mat, n, i, threshold)

    print('----------------Simulation ends----------------\n\n')
    result = {'belief_avg_true_good': belief_avg_true_good, 'confidence_mat': confidence_mat, 'pop': pop}

    return result


