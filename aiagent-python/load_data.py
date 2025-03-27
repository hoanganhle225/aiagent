import minerl
import numpy as np
from tqdm import tqdm

env = minerl.data.make('MineRLNavigateDense-v0')
data_iter = env.batch_iter(32, 32, seq_len=1)

states = []
actions = []

for batch in tqdm(data_iter):
    s, a, r, d, n = batch
    # Lấy các giá trị đơn giản như vị trí + hành động đi/lùi
    states.extend(s['position'])
    actions.extend(a['forward'])

    if len(states) > 10000:
        break

states = np.array(states)
actions = np.array(actions)

np.save("states.npy", states)
np.save("actions.npy", actions)
