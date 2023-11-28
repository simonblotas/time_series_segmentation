import pandas as pd
import numpy as np

from main_constants import databases_path
path_  = databases_path + "/SHOIAB2"
activities_name = {
    0: 'walking',
    1: 'standing',
    2: 'biking',
    3: 'upstairs',
    4: 'jogging',
    5: 'downstairs',
}


def load_shoiab2():
    print('Loading shoiab2 database...')
    activity_recognition_signals = []
    activity_recognition_activities = []
    activity_recognition_activities_indexes = []

    file_paths = {
        "arm":  path_ + "/Arm.xlsx",
        "belt":  path_ + "/Belt.xlsx",
        "wrist": path_ + "/Wrist.xlsx",
        "pocket": path_ + "/Pocket.xlsx"
    }

    for key, path in file_paths.items():
        df = pd.read_excel(path)

        # Extract the desired columns
        timestamp_column = df['Time_Stamp']
        ax_column = df['Ax']
        ay_column = df['Ay']
        az_column = df['Az']
        gx_column = df['Gx']
        gy_column = df['Gy']
        gz_column = df['Gz']
        mx_column = df['Mx']
        my_column = df['My']
        mz_column = df['Mz']
        activity_column = df['Activity_Label']

        # Extract data from the specified columns
        signals = np.array([ax_column, ay_column, az_column], dtype=float)

        # Append data to the activity_recognition_signals array
        activity_recognition_signals.append(signals.T)

        # Find activities
        unique_activities = [activity_column[0]] + [activity_column[i] for i in range(1,len(activity_column)) if activity_column[i] != activity_column[i - 1]] 
        activity_recognition_activities.append(unique_activities)

        # Find indices where the activity changes
        change_indices = [i for i in range(1,len(activity_column)) if activity_column[i] != activity_column[i - 1]] + [len(signals[1])]
        activity_recognition_activities_indexes.append(change_indices)

    print('Loading completed')
    print('Number of signals loaded: ' + str(len(activity_recognition_signals)))

    return activity_recognition_signals, activity_recognition_activities_indexes, activity_recognition_activities
