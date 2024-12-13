import numpy as np
import json
import itertools

# Define ranges for random parameters (log scale!)
lr_min, lr_max = -3, -1# 1e-3, 1e-1 # NOTE that good configs lie <= 0.02
lr_wei_min, lr_wei_max = -5, -2 #1e-5, 1e-2
lr_wix_min, lr_wix_max = -5, -2 #1e-2, 1e0
hidden_layer_width_min, hidden_layer_width_max = 100, 500

# Number of random configurations
num_random_configs = 100

# Generate random configurations
random_configs = []

hp = {
        'lr': [1e-4, 1e-3, 1e-2],
        'weight_decay': [0.5, 0.05, 0.0001],
        'drop': [0, 0.001, 0.0001],
        'layer_decay': [0.65, 0.05, 0.0001],
        'batch_size': [16,32,64],
        'drop_path':[0.01, 0.1, 0.0001]
    }

k, v = zip(*hp.items())
combos = [dict(zip(k, c)) for c in itertools.product(*v)]


# for _ in range(num_random_configs):
#     config = {
#         'lr': 10 ** np.random.uniform(lr_min, lr_max),
#         'lr_wei': 10 ** np.random.uniform(lr_wei_min, lr_wei_max),
#         'lr_wix': 10 ** np.random.uniform(lr_wix_min, lr_wix_max),
#         'hidden_layer_width': int(np.random.uniform(hidden_layer_width_min, hidden_layer_width_max))
#     }
#     random_configs.append(config)

# Save to file
with open('labram_hparams.json', 'w') as f:
    json.dump(combos, f)