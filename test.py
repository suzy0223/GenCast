import numpy as np

data = np.load('/Users/suxs3/Downloads/GenCast/code/Dataset/metr/prompt_embeddings_road_wo_address_limited_poi.npy')
print(data.shape)
# min max 
print(np.min(data), np.max(data))
# mean std
print(np.mean(data), np.std(data))