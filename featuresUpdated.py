import os
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
from scipy.stats import stats, skew, kurtosis

dictionary3D = {}  # dictionary to store all 3D files
# files path
path = 'C:\\Users\\shaher\\Desktop\\Introduction to graduation project\\project\\3D\\3D\\'
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
smooth_Energy = []
dictionary3Dsmooth_Energy = {}
for k in dictionary3Dsmooth.keys():
    smooth_Energy = []
    num_frames = len(dictionary3Dsmooth[k])
    for frame in range(num_frames):
        E_smooth = np.sum((dictionary3Dsmooth[k][0] - dictionary3Dsmooth[k][frame]) ** 2)
        smooth_Energy.append(E_smooth)
    dictionary3Dsmooth_Energy[k] = smooth_Energy

# ---------------------------------------------- THRESHOLD -------------------------------------------------#
thresholds = {}


def thresholdCalculator(k):
    # Get the x and y data
    x_data = np.arange(len(dictionary3Dsmooth_Energy[k]))
    y_data = dictionary3Dsmooth_Energy[k]
    # Perform linear regression
    a, b = np.polyfit(x_data, y_data, deg=1)
    # Generate the best-fit line
    best_fit_line = a * x_data + b
    # Calculate the energy value at x = 0 (y-intercept)
    y_intercept = b
    threshold = y_intercept
    thresholds[k] = threshold


for k in dictionary3Dsmooth_Energy.keys():
    thresholdCalculator(k)

exercisesBefore = []
exercisesBeforeDic = {}
for k in dictionary3Dsmooth_Energy.keys():
    # print("The thresold: ", thresholds[k], " ----- key : ",k)
    exercisesBefore = []
    for count in range(len(dictionary3Dsmooth_Energy[k])):
        if dictionary3Dsmooth_Energy[k][count] > thresholds[k]:
            exercisesBefore.append(1)
        else:
            exercisesBefore.append(0)
    exercisesBeforeDic[k] = exercisesBefore

exercisesAfter = []
exercisesAfterDic = {}
counterOne = 0
counterZero = 0
maxThres = 10

for k in dictionary3Dsmooth_Energy.keys():
    exercisesAfter = []
    counterOne = 0
    counterZero = 0
    for count in range(len(dictionary3Dsmooth_Energy[k])):
        if dictionary3Dsmooth_Energy[k][count] > thresholds[k]:
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
    exercisesAfterDic[k] = exercisesAfter

# -------------------------------------------------- CALCULATION --------------------------------- #
# Assuming 'exercisesAfter' is the array containing the pulses

# Step 1: Detect the beginning and end of each pulse
pulse_starts = []  # To store the indices or timestamps of pulse beginnings
pulse_ends = []  # To store the indices or timestamps of pulse ends
pulse_active = False
pulseStartDic = {}  ####
pulseEndDic = {}  ####
for k in exercisesAfterDic.keys():
    pulse_starts = []  ####
    pulse_ends = []  ####
    for i in range(len(exercisesAfterDic[k])):
        if not pulse_active and exercisesAfterDic[k][i] == 1:
            pulse_active = True
            pulse_starts.append(i)
        elif pulse_active and exercisesAfterDic[k][i] == 0:
            pulse_active = False
            pulse_ends.append(i)
    pulseEndDic[k] = pulse_ends  ####
    pulseStartDic[k] = pulse_starts  ####

# Step 1: Calculate exercise intervals based on pulse positions
exerciseIntervalDic = {}  ####
exercise_intervals = []
first_start = 0
for k in exercisesAfterDic.keys():
    exercise_intervals = []  ####
    for i in range(len(pulseStartDic[k])):  # Iterate up to the second-to-last index
        if i == 0:
            start = first_start
            end = 0
            if i + 1 < len(pulseStartDic[k]):
                end = int((pulseEndDic[k][i] + pulseStartDic[k][i + 1]) / 2)
        elif i == len(pulseStartDic[k]) - 1:
            start = int((pulseStartDic[k][i] + pulseEndDic[k][i - 1]) / 2)
            end = len(exercisesAfterDic[k])
        else:
            start = int((pulseStartDic[k][i] + pulseEndDic[k][i - 1]) / 2)
            end = int((pulseEndDic[k][i] + pulseStartDic[k][i + 1]) / 2)
        exercise_intervals.append((start, end))
    exerciseIntervalDic[k] = exercise_intervals  ####


# print(exerciseIntervalDic)
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


def calculate_sma(x):
    sma = sum(abs(xi) for xi in x) / len(x)
    return sma


def integrand(x):
    integral = np.trapz(np.abs(x), dx=1)
    return integral


def extract_argmax(x):
    Xf = np.fft.fft(x)  # fourier transform
    argmax = [i for i in range(len(Xf)) if all(Xf[i] >= Xf[j] for j in range(len(Xf)))]
    return argmax


def extract_argmin(x):
    Xf = np.fft.fft(x)
    argmin = [i for i in range(len(Xf)) if all(Xf[i] <= Xf[j] for j in range(len(Xf)))]
    return argmin


def extract_argavg_indices(x):
    Xf = np.fft.fft(x)
    average = np.mean(Xf)
    argavg_indices = []
    for i in range(len(Xf)):
        is_argavg = all(abs(Xf[i] - average) <= abs(Xf[j] - average) for j in range(len(Xf)))
        if is_argavg:
            argavg_indices.append(i)
    return argavg_indices


def extract_dc_bias(x):
    Xf = np.fft.fft(x)
    N = len(Xf)
    dc_bias = (1 / N) * np.sum(np.abs(Xf) ** 2)
    return dc_bias


allExercises = {
    'E0': [],
    'E1': [],
    'E2': [],
    'E3': [],
    'E4': [],
    'E5': [],
    'E6': [],
    'E7': [],
    'E8': [],
    'E9': []
}
segments = {}
list = []
for k in exerciseIntervalDic.keys():
    list = []
    print("--------------- key = ", k, " ----------------")
    for i, interval in enumerate(exerciseIntervalDic[k]):
        list = []
        print("Exercise:", i + 1)
        print("Start Point:", interval[0])
        print("End Point:", interval[1])
        print("Num of frames =", interval[1] - interval[0])
        features = {
            'mean': [],  # will store 51 value (17 joint * 3 axis)
            # 'variance': [],
            # 'skewness': [],
            # 'kurtosis': [],
            # 'autocorrelation': [],
            # 'entropy': [],
            # 'sRMS': [],
            # 'sma': [],
            # 'itot': [],
            # 'ARG_MAX': [],
            # 'ARG_MIN': [],
            # 'ARG_AVG': [],
            # 'dc_bias': []
        }

        for joint in range(dictionary3D[k].shape[1]):  # 17 joint
            x_values = []
            y_values = []
            z_values = []
            for j in range(interval[0], interval[1]):
                for axis in range(dictionary3D[k].shape[2]):  # 3 axis
                    if axis == 0:
                        x_values.append(dictionary3D[k][j][joint][axis])
                    elif axis == 1:
                        y_values.append(dictionary3D[k][j][joint][axis])
                    elif axis == 2:
                        z_values.append(dictionary3D[k][j][joint][axis])

            features['mean'].extend([np.mean(x_values), np.mean(y_values), np.mean(z_values)])
            #print(len(features['mean']))
            # features['variance'].extend([np.var(x_values), np.var(y_values), np.var(z_values)])
            # features['skewness'].extend([skew(x_values), skew(y_values), skew(z_values)])
            # features['kurtosis'].extend([kurtosis(x_values), kurtosis(y_values), kurtosis(z_values)])
            # features['autocorrelation'].extend([
            #     np.correlate(x_values, x_values, mode='full'),
            #     np.correlate(y_values, y_values, mode='full'),
            #     np.correlate(z_values, z_values, mode='full')
            # ])
            # features['entropy'].extend(
            #     [calculate_entropy(x_values), calculate_entropy(y_values), calculate_entropy(z_values)])
            # features['sRMS'].extend([calculate_sRMS(x_values), calculate_sRMS(y_values), calculate_sRMS(z_values)])
            # features['sma'].extend([calculate_sma(x_values), calculate_sma(y_values), calculate_sma(z_values)])
            # features['itot'].extend([integrand(x_values), integrand(y_values), integrand(z_values)])
            # features['ARG_MAX'].extend([extract_argmax(x_values), extract_argmax(y_values), extract_argmax(z_values)])
            # features['ARG_MIN'].extend([extract_argmin(x_values), extract_argmin(y_values), extract_argmin(z_values)])
            # features['ARG_AVG'].extend(
            #     [extract_argavg_indices(x_values), extract_argavg_indices(y_values), extract_argavg_indices(z_values)])
            # features['dc_bias'].extend(
            #     [extract_dc_bias(x_values), extract_dc_bias(y_values), extract_dc_bias(z_values)])
        list.append(features)
        #print(" i ", i + 1, " The list : ", list)
        #print("\n")
        # so list containing 10 features[]....
        if k.startswith("E0_"):
            allExercises['E0'].append(list)
            print("all ex :::", len(allExercises['E0']))
            print("+++++++\n\n", allExercises['E0'])
            # allExercises['E0'] contains (10 * 3 * 3 * 10) => 900
        elif k.startswith("E1_"):
            allExercises['E1'].append(list)
        elif k.startswith("E2_"):
            allExercises['E2'].append(list)
        elif k.startswith("E3_"):
            allExercises['E3'].append(list)
        elif k.startswith("E4_"):
            allExercises['E4'].append(list)
        elif k.startswith("E5_"):
            allExercises['E5'].append(list)
        elif k.startswith("E6_"):
            allExercises['E6'].append(list)
        elif k.startswith("E7_"):
            allExercises['E7'].append(list)
        elif k.startswith("E8_"):
            allExercises['E8'].append(list)
        elif k.startswith("E9_"):
            allExercises['E9'].append(list)

        #print(allExercises.keys())
    # allExercises[E0_P0_T0_C0] = segments['E0_P0_T0_C0] = features [8 * 51]
    # print("++++++++++++++++++++++", len(allExercises['E0_P0_T0_C0']))

# print("length of all ex :", len(allExercises))

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
#     print(f"Joint [{joint}] mean(x,y,z): [{np.mean(x_values)}, {np.mean(y_values)}, {np.mean(z_values)}]")
#     print(f"Joint [{joint}] variance(x,y,z): [{np.var(x_values)}, {np.var(y_values)}, {np.var(z_values)}]")
#     print(f"Joint [{joint}] skew(x,y,z): [{skew(x_values)}, {skew(y_values)}, {skew(z_values)}]")
#     print(f"Joint [{joint}] kurtosis(x,y,z): [{kurtosis(x_values)}, {kurtosis(y_values)}, {kurtosis(z_values)}]")
#     print(f"Joint [{joint}] entropy(x,y,z): [{calculate_entropy(x_values)}, {calculate_entropy(y_values)}, {calculate_entropy(z_values)}]")
#     print(f"Joint [{joint}] sRMS(x,y,z): [{calculate_sRMS(x_values)}, {calculate_sRMS(y_values)}, {calculate_sRMS(z_values)}]")
#     print(f"Joint [{joint}] itot(x,y,z): [{integrand(x_values)}, {integrand(y_values)}, {integrand(z_values)}]")
#     print(f"Joint [{joint}] ARG_MAX(x,y,z): [{extract_argmax(x_values)}, {extract_argmax(y_values)}, {extract_argmax(z_values)}]")
#     print(f"Joint [{joint}] ARG_MIN(x,y,z): [{extract_argmin(x_values)}, {extract_argmin(y_values)}, {extract_argmin(z_values)}]")
#     print(f"Joint [{joint}] ARG_AVG(x,y,z): [{extract_argavg_indices(x_values)}, {extract_argavg_indices(y_values)}, {extract_argavg_indices(z_values)}]")
#     print(f"Joint [{joint}] dc_bias(x,y,z): [{extract_dc_bias(x_values)}, {extract_dc_bias(y_values)}, {extract_dc_bias(z_values)}]")
#
# print("---------------------------------------------------------------------------------------------\n")

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

# plt.plot(x_data, y_data, label='Data Points')
# plt.plot(x_data, best_fit_line, label='Best Fit Line')
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
