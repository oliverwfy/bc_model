from model import simulate_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


directory_name = 'consensus/'
bc = np.load(directory_name + 'consensus_bc_evidence_rate_0.3.npy')
ru = np.load(directory_name + 'consensus_ru_evidence_rate_0.3.npy')
eq = np.load(directory_name + 'consensus_eq_evidence_rate_0.3.npy')


evidence_rate_ls = np.linspace(0.001, 0.03, 30)


plt.plot(evidence_rate_ls[0:20], eq[0:20])
plt.plot(evidence_rate_ls[0:20], ru[0:20])
plt.plot(evidence_rate_ls[0:20], bc[0:20])

plt.xlabel(r'evidence rate ($\beta$)', fontsize=14)
plt.ylabel('consensus time', fontsize=14)
# plt.title(rf'$\gamma = {threshold},\alpha = {alpha}, \beta = {prob_evidence}, \epsilon={noise}$', fontsize=14)
plt.legend(['equal weights', 'reliability updating', 'bc'], fontsize=14)

plt.show()