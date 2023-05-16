# how to reverse an array with saving its content..
import pdb
import numpy as np
# Load the data from the .npz file
data = np.load('E0_P0_T0_C0_3D.npz')
array3 = data['reconstruction']
exercise, frames, joints, dimensions = array3.shape
print(exercise,frames, joints, dimensions)
reversed_arr = array3.transpose(0,3,2,1)
pdb.set_trace()