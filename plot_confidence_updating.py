import numpy as np
import matplotlib.pyplot as plt

def confidence_interval(data):
    n = data.shape[1]
    std = data.std(axis=1)
    return 1.960*std/np.sqrt(n)


file_directory = 'fault_detection/'
avg_belief = np.load(file_directory + 'avg_belief.npy')
acc = np.load(file_directory + 'accuracy.npy')
recall = np.load(file_directory + 'recall.npy')
precision = np.load(file_directory + 'precision.npy')


plt.figure()
plt.plot(range(avg_belief.shape[0]), avg_belief.mean(axis=1), color='green')
plt.axhline(y=0.95, color='gray', linestyle='--')
ci = confidence_interval(avg_belief)
plt.fill_between(range(len(avg_belief)), (avg_belief.mean(axis=1)-ci), (avg_belief.mean(axis=1)+ci), alpha=0.2)


plt.xlabel('time step', fontsize=14)
plt.ylabel('average belief', fontsize=14)
plt.title(r'$\gamma = 0.3,\alpha = 0.1, \beta = 0.1, \epsilon=0$', fontsize=14)
plt.legend(['avg belief (95% confidence interval)', 'consensus condition'],fontsize=14)
plt.savefig(file_directory + 'avg_belief.png')


