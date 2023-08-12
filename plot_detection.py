import numpy as np
import matplotlib.pyplot as plt



def confidence_interval_95(data):
    n = data.shape[1]
    std = data.std(axis=1)
    return 1.960*std/np.sqrt(n)

def confidence_interval_90(data):
    n = data.shape[1]
    std = data.std(axis=1)
    return 1.645*std/np.sqrt(n)

file_directory = 'detection/'
acc_individual = np.load(file_directory + 'acc_individual_evidence_rate_0.01.npy')
precision_individual = np.load(file_directory + 'precision_individual_evidence_rate_0.01.npy')
recall_individual = np.load(file_directory + 'recall_individual_evidence_rate_0.01.npy')

acc = np.load(file_directory + 'acc_swarm_evidence_rate_0.01.npy')
precision = np.load(file_directory + 'precision_swarm_evidence_rate_0.01.npy')
recall = np.load(file_directory + 'recall_swarm_evidence_rate_0.01.npy')


plt.figure()
plt.plot(range(acc_individual.shape[0]), acc_individual.mean(axis=1), color='red')
plt.xlabel('time step', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.savefig(file_directory + 'acc_individual.png')


plt.figure()
plt.plot(range(precision_individual.shape[0]), precision_individual.mean(axis=1))
plt.plot(range(recall_individual.shape[0]), recall_individual.mean(axis=1))
plt.xlabel('time step', fontsize=14)
plt.ylabel('precision/recall', fontsize=14)
plt.legend(['precision', 'recall'])
plt.savefig(file_directory + 'precision_recall_individual.png')



plt.figure()
plt.plot(range(acc.shape[0]), acc.mean(axis=1), color='red')
plt.xlabel('time step', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.savefig(file_directory + 'acc_swarm.png')


plt.figure()
plt.plot(range(precision_individual.shape[0]), precision.mean(axis=1))
plt.plot(range(recall.shape[0]), recall.mean(axis=1))
plt.xlabel('time step', fontsize=14)
plt.ylabel('precision/recall', fontsize=14)
plt.legend(['precision', 'recall'])
plt.savefig(file_directory + 'precision_recall_swarm.png')



