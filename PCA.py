import pdb

import numpy as np
from sklearn.decomposition import PCA
import numpy as np

# Load the .npz file
data3 = np.load('E0_P0_T0_C0_3D.npz')
# Access the arrays inside the .npz file by their names
array3 = data3['reconstruction']
frames, joints, dimensions = array3[0].shape
print(frames, joints, dimensions)
reversed_arr = array3[0].transpose(2,1,0)
X = np.array(reversed_arr[0])
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

pdb.set_trace()


