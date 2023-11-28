import pandas as pd
import numpy as np
from main_constants import databases_path


activities_name = {
    0 : 'walking',
    1 : 'standing',
    2 : 'biking',
    3 : 'upstairs',
    4 : 'jogging',
    5 : 'downstairs',
}

def load_shoiab():
    print('Loading shoaib database...')
    shoaib_signals = []
    shoaib_activities = []
    shoaib_activities_indexes = []

    for i in range(1, 11):
        # Read the CSV file
        df = pd.read_csv(databases_path + "/SHOIAB/Participant_" + str(i) +".csv")

        # Extract the desired columns
        left_pocket_columns = df.columns[0:13]
        right_pocket_columns = df.columns[14:27]
        wrist_columns = df.columns[28:41]
        upper_arm_columns = df.columns[42:55]
        belt_columns = df.columns[56:69]
        activity_column = df.columns[69]

        # Extract data from the specified columns
        left_pocket_data = df[left_pocket_columns]
        right_pocket_data = df[right_pocket_columns]
        wrist_data = df[wrist_columns]
        upper_arm_data = df[upper_arm_columns]
        belt_data = df[belt_columns]
        activities = df[activity_column]

        left_pocket_signals = np.array([left_pocket_data['Unnamed: 1'][1:], left_pocket_data['Unnamed: 2'][1:], left_pocket_data['Unnamed: 3'][1:]], dtype=float)
        right_pocket_signals = np.array([right_pocket_data['Unnamed: 15'][1:], right_pocket_data['Unnamed: 16'][1:], right_pocket_data['Unnamed: 17'][1:]], dtype=float)
        wrist_signals = np.array([wrist_data['Unnamed: 29'][1:], wrist_data['Unnamed: 30'][1:], wrist_data['Unnamed: 31'][1:]], dtype=float)
        upper_arm_signals = np.array([upper_arm_data['Unnamed: 43'][1:], upper_arm_data['Unnamed: 44'][1:], upper_arm_data['Unnamed: 45'][1:]], dtype=float)
        belt_signals = np.array([belt_data['Unnamed: 57'][1:], belt_data['Unnamed: 58'][1:], belt_data['Unnamed: 59'][1:]], dtype=float)

        # Append data to the shoaib_signals array
        shoaib_signals.extend([left_pocket_signals.T, right_pocket_signals.T, wrist_signals.T, upper_arm_signals.T, belt_signals.T])

        # Find activities
        unique_activities = [activities[2]] + [activities[i] for i in range(2, len(activities[1:])) if activities[i] != activities[i-1]]

        # Find indices where the activity changes
        change_indices = [i for i in range(2, len(activities[1:])) if activities[i] != activities[i-1]] + [len(left_pocket_signals.T)]
        for i in range(5):
            shoaib_activities.append(unique_activities)
            shoaib_activities_indexes.append(change_indices)
        
    
    print('Loading completed')
    print('Nb_signals : ' + str(len(shoaib_signals)))
    
    return shoaib_signals, shoaib_activities_indexes, shoaib_activities
