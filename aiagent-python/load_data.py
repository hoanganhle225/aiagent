# import minerl
# import numpy as np
# from tqdm import tqdm

# env = minerl.data.make('MineRLTreechop-v0', data_dir='datasets')
# data_iter = env.batch_iter(batch_size=1, num_epochs=1, seq_len=32)

# states = []
# actions = []

# for batch in tqdm(data_iter):
#     s, a, r, d, n = batch
#     # Lấy các giá trị đơn giản như vị trí + hành động đi/lùi
#     states.extend(s['position'])
#     actions.extend(a['forward'])

#     if len(states) > 10000:
#         break

# states = np.array(states)
# actions = np.array(actions)

# np.save("states.npy", states)
# np.save("actions.npy", actions)
if __name__ == "__main__":
    import minerl

    env = minerl.data.make('MineRLTreechop-v0', data_dir='datasets')
    data_iter = env.batch_iter(batch_size=1, num_epochs=1, seq_len=32)

    for obs, action, r, done, _ in data_iter:
        print("Observation:", list(obs.keys()))
        print("Action:", action)
        break
