import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
from scipy.stats import stats, skew, kurtosis

dictionary3D = {}  # dictionary to store all 3D files
# files path
path = 'C:\\Users\\hp\\Desktop\\3D\\3D'
# loop through all files and store them in the dictionary
for npzFile in os.listdir(path):
    f = os.path.join(path, npzFile)
    # checking if it is a file
    if os.path.isfile(f):
        a = np.load(f)
        # load the files into the dictionary
        dictionary3D[npzFile.split('_3D')[0]] = a['reconstruction'][0, :, :, :]
# -------------------------------------------- SEGMENTATION PROCESS ----------------------------------------- #

energies = []
dictionary3D_Energy = {}  # array to store the list of energies for each file..
for k in dictionary3D.keys():
    energies = []
    num_frames = len(dictionary3D[k])  # get the number of frames
    for frame in range(num_frames):
        energy = np.sum(
            (dictionary3D[k][0] - dictionary3D[k][frame]) ** 2)
        energies.append(energy)
    dictionary3D_Energy[k] = energies


# ---------------------------------------------- GAUSSIAN FILTER ----------------------------------------------#
def gkernel_1D(l=3, sig=1):
    """\
    Gaussian Kernel Creator via given length and sigma
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * (np.square(ax)) / np.square(sig))
    return kernel / np.sum(kernel)


# use the gaussian file
b = gkernel_1D(7, 1)
dictionary3Dsmooth = dictionary3D.copy()
for k in dictionary3D.keys():
    for joint_cnt in range(dictionary3D[k].shape[1]):
        for axis_cnt in range(dictionary3D[k].shape[2]):
            dictionary3Dsmooth[k][:, joint_cnt, axis_cnt] = np.convolve(dictionary3D[k][:, joint_cnt, axis_cnt], b)[
                                                            3:-3]  # floor of 5/2
# calculate the energy after filter
dictionary3Dsmooth_Energy = {}
for k in dictionary3Dsmooth.keys():
    dictionary3Dsmooth_Energy[k] = np.square(dictionary3Dsmooth[k]).sum(axis=1).sum(axis=1)

# ---------------------------------------------- THRESHOLD -------------------------------------------------#
threshold = 12.2
exercisesBefore = []
for count in range(dictionary3Dsmooth_Energy['E0_P0_T0_C1'].shape[0]):
    if dictionary3Dsmooth_Energy['E0_P0_T0_C1'][count] > threshold:
        exercisesBefore.append(1)
    else:
        exercisesBefore.append(0)

exercisesAfter = []
counterOne = 0
counterZero = 0
maxThres = 15

print(dictionary3Dsmooth_Energy['E0_P0_T0_C1'].shape[0])
for count in range(dictionary3Dsmooth_Energy['E0_P0_T0_C1'].shape[0]):
    if dictionary3Dsmooth_Energy['E0_P0_T0_C1'][count] > threshold:
        counterOne = counterOne + 1
        if counterOne <= maxThres:
            exercisesAfter.append(0)
            counterZero = counterZero + 1
        else:
            exercisesAfter.append(1)
            counterZero = 0
    else:
        exercisesAfter.append(0)
        counterZero = counterZero + 1
        counterOne = 0

# -------------------------------------------------- CALCULATION --------------------------------- #
# Assuming 'exercisesAfter' is the array containing the pulses

# Step 1: Detect the beginning and end of each pulse
pulse_starts = []  # To store the indices or timestamps of pulse beginnings
pulse_ends = []  # To store the indices or timestamps of pulse ends
pulse_active = False
for i in range(len(exercisesAfter)):
    if not pulse_active and exercisesAfter[i] == 1:
        pulse_active = True
        pulse_starts.append(i)
    elif pulse_active and exercisesAfter[i] == 0:
        pulse_active = False
        pulse_ends.append(i)

# Step 1: Calculate exercise intervals based on pulse positions
exercise_intervals = []
first_start = 0
for i in range(len(pulse_starts)):  # Iterate up to the second-to-last index
    if i == 0:
        start = first_start
        end = int((pulse_ends[i] + pulse_starts[i + 1]) / 2)
    elif i == len(pulse_starts) - 1:
        start = int((pulse_starts[i] + pulse_ends[i - 1]) / 2)
        end = len(exercisesAfter)
    else:
        start = int((pulse_starts[i] + pulse_ends[i - 1]) / 2)
        end = int((pulse_ends[i] + pulse_starts[i + 1]) / 2)
    exercise_intervals.append((start, end))


# ------------------------------------- FEATURES EXTRACTION ----------------------------------------#



def calculate_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_sRMS(signal):
    squared_signal = np.square(signal)
    mean_squared_signal = np.mean(squared_signal)
    sRMS = np.sqrt(mean_squared_signal)
    return sRMS

exercises = {}
for i, interval in enumerate(exercise_intervals):
        print("Exercise:", i + 1)
        print("Start Point:", interval[0])
        print("End Point:", interval[1])
        print("Num of frames =", interval[1] - interval[0])
        features = {
            'mean': [],
            'variance': [],
            'skewness': [],
            'kurtosis': [],
            'autocorrelation': [],
            'entropy': [],
            'sRMS': []
        }

        for joint in range(dictionary3D['E0_P0_T0_C1'].shape[1]):
            x_values = []
            y_values = []
            z_values = []

            for j in range(interval[0], interval[1]):
                for axis in range(dictionary3D['E0_P0_T0_C1'].shape[2]):
                    if axis == 0:
                        x_values.append(dictionary3D['E0_P0_T0_C1'][j][joint][axis])
                    elif axis == 1:
                        y_values.append(dictionary3D['E0_P0_T0_C1'][j][joint][axis])
                    elif axis == 2:
                        z_values.append(dictionary3D['E0_P0_T0_C1'][j][joint][axis])

            features['mean'].extend([np.mean(x_values), np.mean(y_values), np.mean(z_values)])
            features['variance'].extend([np.var(x_values), np.var(y_values), np.var(z_values)])
            features['skewness'].extend([skew(x_values), skew(y_values), skew(z_values)])
            features['kurtosis'].extend([kurtosis(x_values), kurtosis(y_values), kurtosis(z_values)])
            features['autocorrelation'].extend([
                np.correlate(x_values, x_values, mode='full'),
                np.correlate(y_values, y_values, mode='full'),
                np.correlate(z_values, z_values, mode='full')
            ])
            features['entropy'].extend(
                [calculate_entropy(x_values), calculate_entropy(y_values), calculate_entropy(z_values)])
            features['sRMS'].extend([calculate_sRMS(x_values), calculate_sRMS(y_values), calculate_sRMS(z_values)])

            print(f"Joint [{joint}] mean(x,y,z): [{np.mean(x_values)}, {np.mean(y_values)}, {np.mean(z_values)}]")
            print(f"Joint [{joint}] variance(x,y,z): [{np.var(x_values)}, {np.var(y_values)}, {np.var(z_values)}]")
            print(f"Joint [{joint}] skew(x,y,z): [{skew(x_values)}, {skew(y_values)}, {skew(z_values)}]")
            print(f"Joint [{joint}] kurtosis(x,y,z): [{kurtosis(x_values)}, {kurtosis(y_values)}, {kurtosis(z_values)}]")
            print(f"Joint [{joint}] entropy(x,y,z): [{calculate_entropy(x_values)}, {calculate_entropy(y_values)}, {calculate_entropy(z_values)}]")
            print(f"Joint [{joint}] sRMS(x,y,z): [{calculate_sRMS(x_values)}, {calculate_sRMS(y_values)}, {calculate_sRMS(z_values)}]")

        print("---------------------------------------------------------------------------------------------\n")

# -------------------------------------------------- PLOT & FIGURES PART --------------------------------- #
# 1. plot the original signal for one specific exercise before and after filtration
plt.figure(1)
# plot one:
plt.subplot(1, 2, 1)
plt.plot(dictionary3D_Energy['E0_P0_T0_C1'])
# Adding labels and title
plt.xlabel('Frame')
plt.ylabel('Energy')
plt.title('Segmentation')
# plot two:
plt.subplot(1, 2, 2)
plt.plot(dictionary3Dsmooth_Energy['E0_P0_T0_C1'])
plt.xlabel('Frame')
plt.ylabel('Energy')
plt.title('Segmentation')
plt.show()

# 2. plot the exercises before and after operation of threshold
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(exercisesBefore)
plt.title('Before')
# plot two:
plt.subplot(1, 2, 2)
plt.plot(exercisesAfter)
plt.title('After')
plt.show()