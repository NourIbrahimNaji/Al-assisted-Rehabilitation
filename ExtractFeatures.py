import os
import pdb
import numpy as np
from scipy.stats import zscore, skew, kurtosis
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from multiprocessing import Process


import json
import time

start_time = time.time()

#Parameters
a = 20
l = 11
s = 4
# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
dictionary3D = {}  # dictionary to store all 3D files
dictionary2D = {}  # dictionary to store all 2D files
# files path
path = '/AIARUPD/Dataset_CVDLPT_Videos_Segments_npz_old'
# loop through all files and store them in the dictionary
for npzFile in os.listdir(path):
    f = os.path.join(path, npzFile)
    if os.path.isfile(f):
        if "_3D" in f:
            fdata = np.load(f)
            # load the files into the dictionary
            dictionary3D[npzFile.split('_3D')[0]] = fdata['reconstruction'][0, :, :, :]
        # elif "_2D" in f:
        #     fdata = np.load(f)
        #     # load the files into the dictionary
        #     dictionary2D[npzFile.split('_2D')[0]] = fdata['reconstruction'][0, :, :, :]


def window_size(l, s, v):
    w = v - (l - 1) * s
    return w

def sliding_window(data, window_size, step_size):
    for i in range(0, len(data) - window_size + 1, step_size):
        yield data[i:i + window_size]

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
    if np.max(np.abs(x)) < 1e-10:
        return 0.0
    integral = np.trapz(np.abs(x), dx=1)
    return integral

def extract_argmax(x):
    if np.max(np.abs(x)) < 1e-10:
        return 0
    Xf = np.fft.fft(x)
    argmax = np.argmax(np.abs(Xf))
    return argmax

def extract_argmin(x):
    Xf = np.fft.fft(x)  # Fourier transform
    argmin = np.argmin(np.abs(Xf))  # Find the index with the maximum magnitude
    return argmin

def extract_argavg_values(x):
    Xf = np.fft.fft(x)  # Fourier transform
    magnitude = np.abs(Xf)  # Magnitude of the Fourier transform
    average_magnitude = np.mean(magnitude)  # Average magnitude

    # Find the values where the absolute difference from the average is less than or equal for all j
    argavg_values = [i for i, mag_i in enumerate(magnitude) if all(
        abs(mag_i - average_magnitude) <= abs(mag_j - average_magnitude) for j, mag_j in enumerate(magnitude))]

    if not argavg_values:
        return None
    return min(argavg_values)


def extract_dc_bias(x):
    Xf = np.fft.fft(x)
    N = len(Xf)
    dc_bias = (1 / N) * np.sum(np.abs(Xf) ** 2)
    return dc_bias


def featuresExtraction(data, w, s, features,num_autocorrelation_values):
    for window in sliding_window(data, w, s):
        features['mean'].extend([np.mean(window)])
        features['variance'].extend([np.var(window)])
        features['skewness'].extend([skew(window, bias=False)])  # Set bias=False to reduce bias
        features['kurtosis'].extend([kurtosis(window, bias=False)])  # Set bias=False to reduce bias

        autocorrelation = np.correlate(window, window, mode='full')
        #pdb.set_trace()
        # Append 20 equally distributed values from the autocorrelation
        autocorr_len = len(autocorrelation)
        step = autocorr_len // num_autocorrelation_values
        features['autocorrelation'].extend(autocorrelation[:autocorr_len:step][:num_autocorrelation_values])

        features['entropy'].extend(
            [calculate_entropy(window)])
        features['sRMS'].extend([calculate_sRMS(window)])
        features['sma'].extend([calculate_sma(window)])
        features['itot'].extend([integrand(window)])
        features['ARG_MAX'].extend([extract_argmax(window)])
        features['ARG_MIN'].extend([extract_argmin(window)])
        features['ARG_AVG'].extend(
            [extract_argavg_values(window)])
        features['dc_bias'].extend(
            [extract_dc_bias(window)])

allExercises = {f'E{i}': [] for i in range(10)}

i = 0
cnt=0;
for k in dictionary3D.keys():
    cnt = cnt+1
    #if cnt > 10:
    #   break
    v = dictionary3D[k].shape[0]
    w = window_size(l, s, v)
    print(f"file={k}, data length {v}, window size {w}")

    i = i + 1
    features = {
        'mean': [],  # will store 51 value (17 joint * 3 axis)
        'variance': [],
        'skewness': [],
        'kurtosis': [],
        'autocorrelation': [],
        'entropy': [],
        'sRMS': [],
        'sma': [],
        'itot': [],
        'ARG_MAX': [],
        'ARG_MIN': [],
        'ARG_AVG': [],
        'dc_bias': []
    }

    if dictionary3D[k].shape[0] < int(a/2) + 1 + (l - 1) * s:
        dictionary3D[k]=np.concatenate((dictionary3D[k],np.zeros((((l - 1)*s -v + int(a/2) + 1),dictionary3D[k].shape[1],dictionary3D[k].shape[2]))),axis=0)
        w = window_size(l, s, dictionary3D[k].shape[0])
        print(f"........................, new data length {dictionary3D[k].shape[0]}, new window size {w}")

    for joint in range(dictionary3D[k].shape[1]):

        # Extract all axis values for the current joint and frame
        axis_values = dictionary3D[k][:, joint, :]

        # Apply z-score normalization to all axis values
        #axis_values = zscore(axis_values, axis=0)

        # Split the normalized values into x, y, and z components
        x_values = axis_values[:, 0]
        y_values = axis_values[:, 1]
        z_values = axis_values[:, 2]

        # Extract features for x, y, and z values
        #if k == "E0_P1_T0_C0_seg8":
        #pdb.set_trace()

        featuresExtraction(x_values, w, s, features, a)
        featuresExtraction(y_values, w, s, features, a)
        featuresExtraction(z_values, w, s, features, a)

        #for val in features.values():
        #   if np.isnan(val).any():
        #      pdb.set_trace()


    # so list containing 10 features[]....
    print(int(i), " :Done segment: " + k + "\n")
    if k.startswith("E0_"):
        allExercises['E0'].append(features)
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

X, y, Y = [], [], []
# Define dictionaries to store TP, TN, FP, and FN for each exercise
for exercise, features in allExercises.items():
    for feature in features:
        feature_values = [value for key, values in feature.items() for value in values if len(values) > 0]
        if feature_values:  # Check if any non-empty values were appended
            X.append(feature_values)
            y.append(exercise)
            Y.append(int(exercise[1:]))

x=np.array(X)
Y=np.array(Y).reshape(-1,1)
Z=np.concatenate((x,Y),axis=1)
np.save('SavedData',Z)


for  label in np.unique(Z[:,-1]):
     np.save(f"SavedData_E{int(label)}_l{l}_s{s}_a{a}",Z[Z[:,-1]==label])

print("--- %s seconds ---" % (time.time() - start_time))
