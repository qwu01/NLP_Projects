from autofaiss import build_index
import numpy as np

embeddings = np.float32(np.random.rand(100, 512))
index, index_infos = build_index(embeddings, save_on_disk=False)

query = np.float32(np.random.rand(1, 512))
# print(query.shape)
_, I = index.search(query, 1)
print(I)
