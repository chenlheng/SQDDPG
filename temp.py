import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dirt = '/Users/lhchen/nas/SQDDPG/model_save'
models = ['simple_spread_maddpg32bs', 'simple_spread_maddpg1024bs', 'simple_spread_sqddpg']
# models = ['simple_tag_maddpg_128bs', 'simple_tag_maddpg1024bs', 'simple_tag_sqddpg']

file = 'exp.out'

# data = [[] for _ in range(len(models))]
for i, model_name in enumerate(models):
    data = []
    with open('/'.join([dirt, model_name, file]), 'r') as f:
        for j, line in enumerate(f):
            raw_data = line.strip().split()
            if len(raw_data) == 14:
                data.append(float(raw_data[4][:-1]))
    data = np.array(data)
    n = len(data)
    # output_data = data[5000:6000]
    output_data = data
    print(np.mean(output_data[-1000:]))
    df = pd.DataFrame({
        'variable': np.arange(len(output_data) // 100),
        'value': output_data[::100]
    })
    sns.lineplot(x='variable', y='value', label=model_name, data=df)

# plt.ylim(-5, 5)
plt.legend()
plt.show()
