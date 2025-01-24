import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DF_ANALYSIS_PATH = 'src/database/official/'
LOWER, UPPER = 0.025 , 0.050 

an = pd.read_json(DF_ANALYSIS_PATH + 'analysis.json')
combine = []
for name, data in an.items():
    for d in [data[9]]:
        # d = np.array(d)
        # plt.plot(np.arange(len(d)), d[:,0])
        combine.extend(d)
        break
    # break

combine = np.unique(np.array(combine), axis=0)

print(combine)

plt.plot(np.arange(len(combine)), [c[0] for c in combine])
plt.xscale('log')
plt.yscale('log')
plt.show()