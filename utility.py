import pandas as pd
from scipy.stats import beta, norm
from agent import *

tol = 10**-6


def confidence_interval(data_ls):
    df = pd.DataFrame(data_ls)
    l_quantile = df.quantile(0.025, axis=1)
    r_quantile = df.quantile(0.975, axis=1)

    mean = df.mean(axis=1)
    return [mean-l_quantile, r_quantile-mean]


def evidential_update(agent, alpha=0.5, dampening=None):
    factor = (1-alpha)*agent.x
    agent.x = factor / (factor + alpha*(1-agent.x))
    if dampening:
        agent.x = dampening/2 + (1-dampening)*agent.x

    return agent.x


def check_consensus(avg_confidence , ite=-1):

    return avg_confidence[ite][-1] >= 0.99


def my_kl_divergence(x1, x2):

    if x1 == 1.0:
        if x2 == 1.0:
            kl_div = 0
        else:
            if x2 == 0.0:
                kl_div = 1/tol
            else:
                kl_div = x1*np.log(x1/x2)
    else:
        if x2 == 1.0:
            if x1 == 0.0:
                kl_div = 1/tol
            else:
                kl_div = x1*np.log((x1)/x2)

        elif x2 == 0.0:
            kl_div = (1-x1)*np.log((1-x1)/(1-x2))
        else:
            kl_div = x1*np.log((x1)/x2) + (1-x1)*np.log((1-x1)/(1-x2))

    return kl_div

def kl_divergence(x1, x2):
    return entropy([x1, 1-x1], [x2, 1-x2])


# def total_variation_distance(x1, x2):
#     tv_distance = np.abs(x1-x2)
#     return tv_distance


# def softmax(input):
#     output = np.exp(input) / np.exp(input).sum()
#     return output


def weights_rescale(agent, pool_id, euqal_weights = False):

    confidence = agent.confidence[pool_id]

    if not euqal_weights:
        if confidence.sum() < tol:
            return np.zeros((len(confidence)))
        weight = confidence / confidence.sum()
    else:
        if confidence.sum() < tol:
            return 0.0
        return 1.0/confidence.sum()

    return np.array(weight)


def confidence_updating_beta(pool_prob):

    mean = pool_prob.mean()
    var = pool_prob.var()

    a = mean*(mean*(1-mean)/var - 1)
    b = a*(1-mean)/mean

    weight_arr = beta.pdf(pool_prob,a, b)
    return weight_arr / weight_arr.max()


def confidence_updating_norm(pool_prob):
    if np.all(pool_prob == pool_prob[0]):
        return np.ones_like(pool_prob)

    mean = pool_prob.mean()
    std = pool_prob.std()
    weight_arr = norm(mean, std).pdf(pool_prob)

    return weight_arr / weight_arr.max()


def confidence_updating(pool_prob, pooled_prob):
    # return np.exp(-(np.array([my_kl_divergence(pooled_prob, prob) for prob in pool_prob])))
    return np.exp(-(np.array([kl_divergence(pooled_prob, prob) for prob in pool_prob])))


def log_op(x, w):
    numerator = np.prod(x**w)
    return numerator / (numerator + np.prod((1 - x) ** w))

def s_prod(x, w):
    numerator = np.prod(x) ** w
    return numerator / (numerator + np.prod(1-x) ** w)


def opinion_pooling_bc(pool, threshold, model, malicious_type, distance):
    if model == 'bc_own_belief':
        if malicious_type == 'fixed_belief':
            opinion_pooling_own_belief_malicious_1(pool, threshold, distance)
        elif malicious_type == 'min_rule':
            opinion_pooling_own_belief_malicious_2(pool, threshold, distance)
    elif model == 'bc_pooled_belief':
        if malicious_type == 'fixed_belief':
            opinion_pooling_pooled_belief_malicious_1(pool, threshold, distance)
        elif malicious_type == 'min_rule':
            opinion_pooling_pooled_belief_malicious_2(pool, threshold, distance)

    return None


def opinion_pooling_confidence_updating(pool, threshold, model, malicious_type, distance):
    if malicious_type == 'fixed_belief':
        opinion_pooling_confidence_malicious_1(pool, threshold, distance)
    elif malicious_type == 'min_rule':
        opinion_pooling_confidence_malicious_2(pool, threshold, distance)



    return None

def opinion_pooling_own_belief_malicious_1(pool, threshold, distance):
    pool_prob = np.array([agent.x for agent in pool])
    for individual in pool:
        if individual.state:
            if distance == 'kl':
                d = 1 - confidence_updating(pool_prob, individual.x)
            elif distance == 'total_variation':
                d = total_variation(pool_prob, individual.x)
            self_pool = pool_prob[np.where( d < threshold )]
            individual.x = s_prod(self_pool, np.round(1/len(self_pool), 5))

    return None



def opinion_pooling_pooled_belief_malicious_1(pool, threshold, distance):
    pool_prob = np.array([agent.x for agent in pool])
    for individual in pool:
        if individual.state:
            pooled_belief = s_prod(pool_prob, np.round(1/len(pool_prob), 5))
            if distance == 'kl':
                d = 1 - confidence_updating(pool_prob, pooled_belief)
            elif distance == 'total_variation':
                d = total_variation(pool_prob, pooled_belief)

            self_pool = pool_prob[np.where( d < threshold )]
            individual.x = s_prod(self_pool, np.round(1/len(self_pool), 5)) if self_pool.size != 0 else pooled_belief

    return None


def opinion_pooling_confidence_malicious_1(pool, threshold, distance):
    pool_id = np.array([agent.id for agent in pool])
    pool_prob = np.array([agent.x for agent in pool])

    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id)

            # log linear
            pooled_prob = log_op(pool_prob, weights)

            # new confidence
            if distance == 'kl':
                confidence_new = confidence_updating(pool_prob, pooled_prob)
            elif distance == 'total_variation':
                confidence_new = 1 - total_variation(pool_prob, pooled_prob)

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():
                mal_id = pool_id[np.where(confidence_new < threshold)[0]]
                confidence_new[confidence_new < threshold] = 0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))

                if distance == 'kl':
                    confidence_new = confidence_updating(pool_prob, pooled_prob)
                elif distance == 'total_variation':
                    confidence_new = 1 - total_variation(pool_prob, pooled_prob)

                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
                individual.confidence[mal_id] = 0

            else:
                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new


    return


def opinion_pooling_own_belief_malicious_2(pool, threshold, distance):

    pool_prob = np.array([agent.x for agent in pool])
    pool_normal_belief = [agent.x for agent in pool if agent.state]
    min_belief = np.min(pool_normal_belief) if pool_normal_belief else 0.5

    for individual in pool:
        if individual.state:
            if distance == 'kl':
                d = 1 - confidence_updating(pool_prob, individual.x)
            elif distance == 'total_variation':
                d = total_variation(pool_prob, individual.x)

            self_pool = pool_prob[np.where( d < threshold )]
            individual.x = s_prod(self_pool, np.round(1/len(self_pool), 5))
        else:
            individual.x = min_belief

    return None


def opinion_pooling_pooled_belief_malicious_2(pool, threshold, distance):

    pool_prob = np.array([agent.x for agent in pool])
    pool_normal_belief = [agent.x for agent in pool if agent.state]
    min_belief = np.min(pool_normal_belief) if pool_normal_belief else 0.5

    for individual in pool:
        if individual.state:
            pooled_belief = s_prod(pool_prob, np.round(1/len(pool_prob), 5))

            if distance == 'kl':
                d = 1 - confidence_updating(pool_prob, pooled_belief)
            elif distance == 'total_variation':
                d = total_variation(pool_prob, pooled_belief)

            self_pool = pool_prob[np.where( d < threshold )]
            individual.x = s_prod(self_pool, np.round(1/len(self_pool), 5)) if self_pool.size != 0 else pooled_belief

        else:
            individual.x = min_belief

    return None


def opinion_pooling_confidence_malicious_2(pool, threshold, distance):
    pool_prob = np.array([agent.x for agent in pool])
    pool_normal_belief = [agent.x for agent in pool if agent.state]
    min_belief = np.min(pool_normal_belief) if pool_normal_belief else 0.5
    pool_id = np.array([agent.id for agent in pool])

    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id)

            # log linear
            pooled_prob = log_op(pool_prob, weights)

            # new confidence

            if distance == 'kl':
                confidence_new = confidence_updating(pool_prob, pooled_prob)
            elif distance == 'total_variation':
                confidence_new = 1 - total_variation(pool_prob, pooled_prob)

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():
                mal_id = pool_id[np.where(confidence_new<threshold)[0]]
                confidence_new[confidence_new<threshold] = 0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))

                if distance == 'kl':
                    confidence_new = confidence_updating(pool_prob, pooled_prob)
                elif distance == 'total_variation':
                    confidence_new = 1 - total_variation(pool_prob, pooled_prob)

                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
                individual.confidence[mal_id] = 0

            else:
                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
        else:
            individual.x = min_belief

    return










def generate_malicious_agents(pop, malicious, mal_c):
    malicious_id = []
    if malicious:
        malicious_id = np.arange(0, int(len(pop)*malicious))
        for i in malicious_id:
            pop[i] = Agent(len(pop), pop[i].id, mal_c, False)

    return malicious_id


def generate_malfunctioning_agents(pop, malfunctioning, init_x=None):

    malfunctioning_id = []
    if malfunctioning:

        # malfunctioning_id = np.arange(0, len(pop), int(1/malfunctioning))
        malfunctioning_id = np.arange(0, int(len(pop)*malfunctioning))
        malfunctioning_agents = pop[malfunctioning_id]

        for mal_agent in malfunctioning_agents:
            mal_agent.state = False
            if not init_x:
                mal_agent.x = np.random.random()
            else:
                mal_agent.x = init_x

    return malfunctioning_id


def avg_belief_good(mal_id, pop):

    good_ids = list(set(range(len(pop))) - set(mal_id))
    good_agents = pop[good_ids]
    belief_avg = np.sum([agent.x for agent in good_agents]) / len(good_agents)

    return belief_avg


def opinion_pooling_own_belief_malicious_1_total_variation(pop, confidence_mat, n, i, threshold=0.5):
    pool_prob = np.array([agent.x for agent in pop])
    for individual in pop:
        if individual.state:
            d = total_variation(pool_prob, individual.x)
            confidence_mat[n,i,individual.id,:] = d
            self_pool = pool_prob[np.where( d < threshold )]
            individual.x = s_prod(self_pool, np.round(1/len(self_pool), 5)) if self_pool.size != 0 else individual.x

    return None

def total_variation(pool_prob, own):
    return np.round([np.abs(x-own) for x in pool_prob], 6)
