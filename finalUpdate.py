import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
testFile = 'E9_P4_T2_C0'
dictionary3D = {}  # dictionary to store all 3D files
# files path
path = 'C:\\Users\\shaher\\Desktop\\Introduction to graduation project\\project\\3D\\3D_blurred\\'
# loop through all files and store them in the dictionary
for npzFile in os.listdir(path):
    f = os.path.join(path, npzFile)
    # checking if it is a file
    if os.path.isfile(f):
        a = np.load(f)
        # load the files into the dictionary
        dictionary3D[npzFile.split('_a')[0]] = a['reconstruction'][0, :, :, :]

# ------------------------------------------- Energy calculation before filter ---------------------------- #
dictionary3D_Energy = {}  # array to store the list of energies for each file..
for k in dictionary3D.keys():
    dictionary3D_Energy[k] = np.square(dictionary3D[k] - dictionary3D[k][0]).sum(axis=1).sum(axis=1)

for k in dictionary3D.keys():
    while dictionary3D_Energy[k][dictionary3D_Energy[k].argmax()] > 10:
        dictionary3D_Energy[k][dictionary3D_Energy[k].argmax()] = dictionary3D_Energy[k][
            dictionary3D_Energy[k].argmax() - 1]


# ------------------------------------------- GAUSSIAN FILTER ------------------------------------------- #
def gkernel_1D(l=3, sig=1):  # length and sigma
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * (np.square(ax)) / np.square(sig))
    return kernel / np.sum(kernel)


# use the gaussian filter
b = gkernel_1D(7, 1)
dictionary3Dsmooth = dictionary3D.copy()
for k in dictionary3D.keys():
    for joint_cnt in range(dictionary3D[k].shape[1]):
        for axis_cnt in range(dictionary3D[k].shape[2]):
            dictionary3Dsmooth[k][:, joint_cnt, axis_cnt] = np.convolve(dictionary3D[k][:, joint_cnt, axis_cnt], b)[
                                                            3:-3]

# ------------------------------------------- Energy calculation after filter ---------------------------- #
dictionary3Dsmooth_Energy = {}
for k in dictionary3Dsmooth.keys():
    dictionary3Dsmooth_Energy[k] = np.square(dictionary3Dsmooth[k] - dictionary3Dsmooth[k][0]).sum(axis=1).sum(axis=1)[
                                   3:-3]

# ------------------------------------------- NORMALIZATION ------------------------------------------- #
# normalization process to make the energy in the range from (1 to 0)
dictionary3Dnormalized_Energy = {}
for k in dictionary3Dsmooth.keys():
    dictionary3Dnormalized_Energy[k] = (dictionary3Dsmooth_Energy[k] - np.min(dictionary3Dsmooth_Energy[k])) / (
            np.max(dictionary3Dsmooth_Energy[k]) - np.min(dictionary3Dsmooth_Energy[k]))

# ------------------------------------------- SEGMENTATION ------------------------------------------- #
Thresholds = [0.28, 0.28, 0.419, 0.42, 0.2885, 0.28, 0.28, 0.4, 0.4, 0.285]
MoveWindowSize = [25, 25, 35, 20, 26, 30, 30, 33, 28, 35]
MaskWindow = [0.62, 0.62, 0.5, 0.45, 0.6533, 0.73, 0.73, 0.22, 0.37, 0.6]

thresholdValue = 0
dictionary3Dnormalized_Mask = {}
for k in dictionary3Dsmooth.keys():
    if k.startswith("E0_"):
        thresholdValue = Thresholds[0]
    elif k.startswith("E1_"):
        thresholdValue = Thresholds[1]
    elif k.startswith("E2_"):
        thresholdValue = Thresholds[2]
    elif k.startswith("E3_"):
        thresholdValue = Thresholds[3]
    elif k.startswith("E4_"):
        thresholdValue = Thresholds[4]
    elif k.startswith("E5_"):
        thresholdValue = Thresholds[5]
    elif k.startswith("E6_"):
        thresholdValue = Thresholds[6]
    elif k.startswith("E7_"):
        thresholdValue = Thresholds[7]
    elif k.startswith("E8_"):
        thresholdValue = Thresholds[8]
    elif k.startswith("E9_"):
        thresholdValue = Thresholds[9]
    dictionary3Dnormalized_Mask[k] = dictionary3Dnormalized_Energy[k] > thresholdValue


# ------------------------------------------- MOVING AVERAGE FUNCTION ------------------------------------------- #

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


moveValue = 0
dictionary3Dnormalized_MaskMovingAverage = {}
for k in dictionary3Dnormalized_Mask.keys():
    if k.startswith("E0_"):
        moveValue = MoveWindowSize[0]
    elif k.startswith("E1_"):
        moveValue = MoveWindowSize[1]
    elif k.startswith("E2_"):
        moveValue = MoveWindowSize[2]
    elif k.startswith("E3_"):
        moveValue = MoveWindowSize[3]
    elif k.startswith("E4_"):
        moveValue = MoveWindowSize[4]
    elif k.startswith("E5_"):
        moveValue = MoveWindowSize[5]
    elif k.startswith("E6_"):
        moveValue = MoveWindowSize[6]
    elif k.startswith("E7_"):
        moveValue = MoveWindowSize[7]
    elif k.startswith("E8_"):
        moveValue = MoveWindowSize[8]
    elif k.startswith("E9_"):
        moveValue = MoveWindowSize[9]
    dictionary3Dnormalized_MaskMovingAverage[k] = moving_average(dictionary3Dnormalized_Mask[k], moveValue)

# ------------------------------------------- MASK AFTER MOVING AVERAGE ------------------------------------------- #
maskValue = 0
dictionary3Dnormalized_MaskModified = {}
for k in dictionary3Dnormalized_Mask.keys():
    if k.startswith("E0_"):
        maskValue = MaskWindow[0]
    elif k.startswith("E1_"):
        maskValue = MaskWindow[1]
    elif k.startswith("E2_"):
        maskValue = MaskWindow[2]
    elif k.startswith("E3_"):
        maskValue = MaskWindow[3]
    elif k.startswith("E4_"):
        maskValue = MaskWindow[4]
    elif k.startswith("E5_"):
        maskValue = MaskWindow[5]
    elif k.startswith("E6_"):
        maskValue = MaskWindow[6]
    elif k.startswith("E7_"):
        maskValue = MaskWindow[7]
    elif k.startswith("E8_"):
        maskValue = MaskWindow[8]
    elif k.startswith("E9_"):
        maskValue = MaskWindow[9]
    dictionary3Dnormalized_MaskModified[k] = dictionary3Dnormalized_MaskMovingAverage[k] > maskValue

# ----------------- CALCULATION TIME FOR EACH EXERCISE----------------- #
# assuming 'exercisesAfterDic' is the dictionary containing the pulses for each key[k]

# step 1: Detect the beginning and end of each pulse
pulse_starts = []  # to store the indices or timestamps of pulse beginnings
pulse_ends = []  # to store the indices or timestamps of pulse ends
pulse_active = False
# two dictionaries to store the start and end of each key
pulseStartDic = {}
pulseEndDic = {}
for k in dictionary3Dnormalized_MaskModified.keys():
    pulse_starts = []
    pulse_ends = []
    for i in range(len(dictionary3Dnormalized_MaskModified[k])):
        if not pulse_active and dictionary3Dnormalized_MaskModified[k][i] == 1:
            pulse_active = True
            pulse_starts.append(i)
        elif pulse_active and dictionary3Dnormalized_MaskModified[k][i] == 0:
            pulse_active = False
            pulse_ends.append(i)
    pulseEndDic[k] = pulse_ends
    pulseStartDic[k] = pulse_starts

# step 2: Calculate exercise intervals based on pulse positions
exerciseIntervalDic = {}
exercise_intervals = []
first_start = 0
for k in dictionary3Dnormalized_MaskModified.keys():
    exercise_intervals = []
    for i in range(len(pulseStartDic[k])):  # iterate up to the second-to-last index
        if i == 0:
            start = first_start
            end = 0
            if i + 1 < len(pulseStartDic[k]):
                end = int((pulseEndDic[k][i] + pulseStartDic[k][i + 1]) / 2)
        elif i == len(pulseStartDic[k]) - 1:
            start = int((pulseStartDic[k][i] + pulseEndDic[k][i - 1]) / 2)
            end = len(dictionary3Dnormalized_MaskModified[k])
        else:
            start = int((pulseStartDic[k][i] + pulseEndDic[k][i - 1]) / 2)
            end = int((pulseEndDic[k][i] + pulseStartDic[k][i + 1]) / 2)
        exercise_intervals.append((start, end))
    exerciseIntervalDic[k] = exercise_intervals


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(dictionary3Dnormalized_MaskModified[testFile])
plt.title("Masked Energies")
plt.xlabel("Frame\n(1)")
plt.ylabel("Energy")

# plot 2:
plt.subplot(1, 2, 2)
plt.plot(dictionary3Dnormalized_MaskModified[testFile])
plt.title("Masked Energies")
plt.xlabel("Frame\n(2)")
plt.ylabel("Energy")
plt.show()
# ----------------- FEATURES EXTRACTION -----------------#
tens = 0
to_remove = []
for k in exerciseIntervalDic.keys():
    numEx = 0
    for i, interval in enumerate(exerciseIntervalDic[k]):
        numEx = numEx + 1
    if ((numEx < 10) | (numEx > 10)):
        tens = tens + 1
        to_remove.append(k)

for k in to_remove:
    del exerciseIntervalDic[k]

for k in exerciseIntervalDic.keys():
    c = 0
    for i, interval in enumerate(exerciseIntervalDic[k]):
        c = c + 1


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

list = []
temp = 0
for k in exerciseIntervalDic.keys():
    #if (k.startswith("E9")):
        list = []
        print("--------------- key = ", k, " ----------------")
        for i, interval in enumerate(exerciseIntervalDic[k]):
            # print("Exercise:", i + 1)
            # print("Start Point:", interval[0])
            # print("End Point:", interval[1])
            # print("Num of frames =", interval[1] - interval[0])
            features = {
                'mean': [],  # will store 51 value (17 joint * 3 axis)
                'variance': [],
                'skewness': [],
                'kurtosis': [],
                #'autocorrelation': []
                'entropy': [],
                'sRMS': [],
                'sma': [],
                'itot': []
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
                x_values = zscore(x_values, axis=0)
                y_values = zscore(y_values, axis=0)
                z_values = zscore(z_values, axis=0)
                features['mean'].extend([np.mean(x_values), np.mean(y_values), np.mean(z_values)])
                features['variance'].extend([np.var(x_values), np.var(y_values), np.var(z_values)])
                features['skewness'].extend([skew(x_values), skew(y_values), skew(z_values)])
                features['kurtosis'].extend([kurtosis(x_values), kurtosis(y_values), kurtosis(z_values)])
                # autocorrelation_x = np.correlate(x_values, x_values, mode='full')
                # autocorrelation_y = np.correlate(y_values, y_values, mode='full')
                # autocorrelation_z = np.correlate(z_values, z_values, mode='full')
                #
                # # Concatenate the autocorrelation arrays
                # autocorrelation = np.concatenate([autocorrelation_x, autocorrelation_y, autocorrelation_z])
                #
                # # Append the concatenated autocorrelation array to the features dictionary
                # features['autocorrelation'].append(autocorrelation)

                features['entropy'].extend(
                    [calculate_entropy(x_values), calculate_entropy(y_values), calculate_entropy(z_values)])
                features['sRMS'].extend([calculate_sRMS(x_values), calculate_sRMS(y_values), calculate_sRMS(z_values)])
                features['sma'].extend([calculate_sma(x_values), calculate_sma(y_values), calculate_sma(z_values)])
                features['itot'].extend([integrand(x_values), integrand(y_values), integrand(z_values)])
                # features['ARG_MAX'].extend([extract_argmax(x_values), extract_argmax(y_values), extract_argmax(z_values)])
                # features['ARG_MIN'].extend([extract_argmin(x_values), extract_argmin(y_values), extract_argmin(z_values)])
                # features['ARG_AVG'].extend(
                #     [extract_argavg_indices(x_values), extract_argavg_indices(y_values), extract_argavg_indices(z_values)])
                # features['dc_bias'].extend(
                #     [extract_dc_bias(x_values), extract_dc_bias(y_values), extract_dc_bias(z_values)])
            # so list containing 10 features[]....
            #print(features)
            if k.startswith("E0_"):
                allExercises['E0'].append(features)
                # allExercises['E0'] contains (10 * 3 * 3 * 10) => 900
            elif k.startswith("E1_"):
                allExercises['E1'].append(features)
            elif k.startswith("E2_"):
                allExercises['E2'].append(features)
            elif k.startswith("E3_"):
                allExercises['E3'].append(features)
            elif k.startswith("E4_"):
                allExercises['E4'].append(features)
            elif k.startswith("E5_"):
                allExercises['E5'].append(features)
            elif k.startswith("E6_"):
                allExercises['E6'].append(features)
            elif k.startswith("E7_"):
                allExercises['E7'].append(features)
            elif k.startswith("E8_"):
                allExercises['E8'].append(features)
            elif k.startswith("E9_"):
                allExercises['E9'].append(features)

# -------------------------------------------------------- RANDOM FOREST ----------------------------------------- #

X = []
y = []

for exercise, features in allExercises.items():
    for feature in features:
        feature_values = []
        for key, value in feature.items():
            if len(value) > 0:  # Check if the value is not empty
                feature_values.extend(value)
        if len(feature_values) > 0:  # Check if any non-empty values were appended
            X.append(feature_values)
            y.append(exercise)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Prepare the feature for the new data point
new_data = [{'mean': [-9.356543e-08, -2.3218089e-08, -7.2357267e-07, -1.2059544e-07, -7.00008e-08, -1.7413566e-06, -1.9406164e-08, -5.5446183e-08, -6.126803e-07, 8.316928e-09, -6.376311e-08, 1.2336776e-07, -7.596127e-07, -6.154526e-07, -1.8435856e-07, 7.2080034e-08, 4.4356945e-08, -3.5762787e-07, 1.3307084e-07, 1.1019929e-07, 1.3584314e-07, 4.6158945e-07, -2.0792317e-08, -1.5271524e-07, 1.1920929e-07, 6.9307726e-10, -3.6733095e-07, 1.0118928e-07, -2.7723091e-08, -8.0396966e-08, 1.9614086e-07, 1.14357746e-07, 1.9322995e-06, -2.800032e-07, 4.629756e-07, 1.9198241e-07, 3.9089556e-07, 1.02575434e-07, 1.0111997e-06, 4.9901564e-07, -8.0396966e-08, -5.486573e-07, -1.5533594e-07, 6.029772e-08, -9.460505e-07, -2.596874e-07, 2.3564628e-08, -1.0271405e-06, 5.4822414e-07, 2.9109245e-07, -1.9040132e-06], 'variance': [1.0000001, 1.0, 0.9999998, 1.0, 0.99999994, 0.99999994, 1.0, 1.0, 1.0, 0.99999994, 1.0000001, 0.99999994, 1.0, 1.0000002, 1.0000001, 0.99999994, 1.0, 1.0, 1.0, 0.99999994, 0.99999994, 0.99999994, 0.9999998, 1.0000001, 1.0000001, 1.0, 0.99999994, 1.0000001, 1.0000001, 0.99999994, 1.0, 0.99999994, 0.9999998, 1.0000001, 0.99999994, 1.0, 1.0, 0.99999976, 1.0, 1.0, 0.99999994, 0.99999994, 1.0000001, 1.0000001, 1.0, 1.0, 1.0000001, 1.0, 1.0, 1.0000001, 1.0000001], 'skewness': [0.7507767526906535, 2.105646848678589, -6.008973302863756, 0.11138352751731873, -0.37726784145246467, -5.8211288888271575, -0.04471752792596817, -0.5342383980751038, -0.8763427734375, 0.15234804025423868, -0.2848876503310814, 0.27878283966626244, -0.06919562816619873, 0.25160768570624564, -4.311358157749482, 1.4275778616289354, -0.05788041651248932, -0.5671635270118713, 1.2550996541976929, 0.34605154508813085, -0.5932426386990515, 0.42913288734551197, -0.7162262334238235, -5.61756891890125, 0.7935215603968648, -0.33163684606552124, -5.764344730765639, 0.8210551462670855, -0.40277792761720305, -5.313560960909207, 0.2885398864746094, -0.30560657134114805, -5.439478425839652, 0.06463799214875804, -0.942315841524228, -5.655858039855957, -0.5380098819732666, -0.18078355996020407, -5.209030628204346, -0.23788827657699585, -0.0016948194594147295, -5.68162110581586, 1.4321871819161371, -0.32895654518189044, -5.918141841888428, 0.5102865099906921, -0.5346076724634727, -5.451247215270996, 1.161138892173767, -0.22246855989660624, -5.435507279245715], 'kurtosis': [0.20005793570086317, 4.429549217224121, 35.10340336682795, -0.3783445358276367, -1.5596173476716746, 33.51413399149572, -1.7369171380996704, -0.7656269073486328, 1.9394893646240234, -1.6472004231453494, -1.2298657346092154, -1.2910700191342914, -1.3738515377044678, -0.8371165575426578, 22.254609762522872, 0.7758431167415387, -0.33304524421691895, -0.719003438949585, 0.0945274829864502, -0.5512473048266622, -1.3287252579599986, -0.1160723109331605, -0.8331176637832609, 31.879035263291037, 0.3622624010413644, -1.040168046951294, 33.081821928241354, 0.7344248928010191, -1.1305676387387449, 29.469425257375754, -0.24809575080871582, -1.011666299303336, 30.665588974566205, -0.8738856616293109, 0.0549625223830259, 32.157955169677734, -0.8601622581481934, -0.7620652971580335, 28.63704490661621, 1.308448314666748, -1.185680053911222, 32.57719464030189, 1.6717193909359764, -1.456243525447027, 34.48952865600586, -0.3236098289489746, -1.2343496717573246, 30.74976348876953, 3.533660411834717, -1.2854857956205146, 30.781520418757815], 'entropy': [5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098, 5.426264754702098], 'sRMS': [1.0, 1.0, 1.0, 1.0, 0.99999994, 1.0, 1.0, 1.0, 0.99999994, 0.99999994, 1.0, 0.99999994, 1.0, 1.0, 1.0, 0.9999999, 1.0, 1.0, 1.0, 1.0, 0.99999994, 1.0, 0.9999999, 1.0, 0.99999994, 1.0, 1.0, 1.0, 1.0, 0.99999994, 0.9999999, 0.99999994, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999999, 0.9999999, 0.99999994, 0.99999994, 0.9999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99999994, 1.0, 1.0], 'sma': [0.8091865419301876, 0.6443826772034341, 0.34631811584844147, 0.7924971528524576, 0.9268616976779561, 0.35061231801329656, 0.9453296203079612, 0.829320601079353, 0.8146079443395138, 0.9264953594741433, 0.8651465201333468, 0.8717393662000812, 0.9117467779406282, 0.8511706760889569, 0.5183241556636816, 0.805964408442378, 0.8218509502882181, 0.8270353838952995, 0.8007772732959237, 0.81163773113905, 0.9196409914902476, 0.7805853422384622, 0.8856265491177869, 0.3643153250910515, 0.8012307025059018, 0.848348157415383, 0.3555861598335553, 0.7493644098989493, 0.8555330254657324, 0.4063069552269786, 0.8047581645118635, 0.8448104900416247, 0.41131720735236654, 0.8573164487647456, 0.8081522979015527, 0.38532242849252596, 0.845403402004131, 0.8282842063453308, 0.43514364616956214, 0.7210249567274437, 0.8784185699078926, 0.3876212900287883, 0.7381030080276866, 0.9112952899101169, 0.34065513206689163, 0.814181511441982, 0.8886608842890277, 0.3963482298043578, 0.7535492882435751, 0.8728402030849179, 0.3766427919225291], 'itot': [34.70175, 27.28793, 11.618802, 32.968178, 39.493404, 11.831888, 39.410275, 34.11383, 32.8226, 38.70283, 35.825462, 36.58392, 37.669704, 35.3744, 19.277067, 32.92422, 33.85201, 33.67982, 32.727943, 33.806267, 37.87565, 32.148834, 37.2747, 12.548752, 33.855125, 36.281857, 11.920117, 31.863438, 36.468765, 14.240086, 34.341152, 35.778404, 14.4468355, 35.893356, 33.45339, 13.29019, 34.982555, 34.134537, 15.476104, 29.082624, 37.109886, 13.517294, 31.589262, 38.843258, 11.290188, 33.973145, 37.648785, 13.628119, 30.304178, 36.88537, 13.096088]}]
new_X = []
for feature in new_data:
    feature_values = []
    for key, value in feature.items():
        feature_values.extend(value)
    new_X.append(feature_values)

# Make predictions on the new data point
prediction = rf_classifier.predict(new_X)

print("Prediction:", prediction)

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1)
plt.plot(dictionary3Dnormalized_Energy[testFile])

# plot 2:
plt.subplot(1, 4, 2)
plt.plot(dictionary3Dnormalized_Mask[testFile])

# plot 3:
plt.subplot(1, 4, 3)
plt.plot(dictionary3Dnormalized_MaskMovingAverage[testFile])

plt.subplot(1, 4, 4)
plt.plot(dictionary3Dnormalized_MaskModified[testFile])
#plt.show()


