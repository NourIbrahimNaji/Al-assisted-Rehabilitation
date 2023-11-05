import os
import numpy as np
from scipy.stats import zscore, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #

testFile = 'E0_P0_T0_C0_seg7'
dictionary3D = {}  # dictionary to store all 3D files
dictionary2D = {}  # dictionary to store all 2D files
# files path
path = 'D:\\Dataset_CVDLPT_Videos_Segments_npz\\'
# loop through all files and store them in the dictionary
for npzFile in os.listdir(path):
    f = os.path.join(path, npzFile)
    # checking if it is a file
    if os.path.isfile(f):
        if "_3D" in f:
            a = np.load(f)
            # load the files into the dictionary
            dictionary3D[npzFile.split('_3D')[0]] = a['reconstruction'][0, :, :, :]
        elif "_2D" in f:
            a = np.load(f)
            # load the files into the dictionary
            dictionary2D[npzFile.split('_2D')[0]] = a['reconstruction'][0, :, :, :]


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
    integral = np.trapz(np.abs(x), dx=1)
    return integral


def extract_argmax(x):
    Xf = np.fft.fft(x)  # Fourier transform
    argmax = np.argmax(np.abs(Xf))  # Find the index with the maximum magnitude
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

i = 0
for k in dictionary3D.keys():
    l = 11
    s = 4
    v = dictionary3D[k].shape[0]
    print(k)
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

    for joint in range(dictionary3D[k].shape[1]):  # 17 joint
        x_values = []
        y_values = []
        z_values = []
        for frame in range(0, dictionary3D[k].shape[0]):
            for axis in range(dictionary3D[k].shape[2]):  # 3 axis
                if axis == 0:
                    x_values.append(dictionary3D[k][frame][joint][axis])
                elif axis == 1:
                    y_values.append(dictionary3D[k][frame][joint][axis])
                elif axis == 2:
                    z_values.append(dictionary3D[k][frame][joint][axis])
        x_values = zscore(x_values, axis=0)
        y_values = zscore(y_values, axis=0)
        z_values = zscore(z_values, axis=0)
        features['mean'].extend([np.mean(x_values), np.mean(y_values), np.mean(z_values)])
        features['variance'].extend([np.var(x_values), np.var(y_values), np.var(z_values)])
        features['skewness'].extend([skew(x_values), skew(y_values), skew(z_values)])
        features['kurtosis'].extend([kurtosis(x_values), kurtosis(y_values), kurtosis(z_values)])
        autocorrelation_x = np.correlate(x_values, x_values, mode='full')
        autocorrelation_y = np.correlate(y_values, y_values, mode='full')
        autocorrelation_z = np.correlate(z_values, z_values, mode='full')
        autocorrelation = np.concatenate([autocorrelation_x, autocorrelation_y, autocorrelation_z])
        # Append the concatenated autocorrelation array to the features dictionary
        features['autocorrelation'].extend(autocorrelation)
        features['entropy'].extend(
            [calculate_entropy(x_values), calculate_entropy(y_values), calculate_entropy(z_values)])
        features['sRMS'].extend([calculate_sRMS(x_values), calculate_sRMS(y_values), calculate_sRMS(z_values)])
        features['sma'].extend([calculate_sma(x_values), calculate_sma(y_values), calculate_sma(z_values)])
        features['itot'].extend([integrand(x_values), integrand(y_values), integrand(z_values)])
        features['ARG_MAX'].extend([extract_argmax(x_values), extract_argmax(y_values), extract_argmax(z_values)])
        features['ARG_MIN'].extend([extract_argmin(x_values), extract_argmin(y_values), extract_argmin(z_values)])
        features['ARG_AVG'].extend(
            [extract_argavg_values(x_values), extract_argavg_values(y_values), extract_argavg_values(z_values)])
        features['dc_bias'].extend(
            [extract_dc_bias(x_values), extract_dc_bias(y_values), extract_dc_bias(z_values)])
    # so list containing 10 features[]....
    print(int(i), " :Done segment: " + k + "\n")
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
v = []

for exercise, features in allExercises.items():
    for feature in features:
        for key, value in feature.items():
            v.append(len(value))
        max_length = max(v)

for exercise, features in allExercises.items():
    for feature in features:
        feature_values = []
        for key, value in feature.items():
            if len(value) > 0:  # Check if the value is not empty
                while len(value) < max_length:
                    value.append(0)  # You can choose any appropriate padding value
                feature_values.extend(value)

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
print("Accuracy: ", accuracy)
