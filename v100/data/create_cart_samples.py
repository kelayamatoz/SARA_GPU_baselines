import numpy as np

n_features = 256
n_samples = 102400

raw_mat = np.random.rand(n_features, n_samples)
spatial_mat_file = 'cart_samples.spatial'
cuda_mat_file = 'cart_samples.cuda'

np.savetxt(spatial_mat_file, raw_mat, delimiter=',')
np.savetxt(cuda_mat_file, raw_mat.T, delimiter=',')
