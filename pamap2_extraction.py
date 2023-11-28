import pandas as pd
import numpy as np
from main_constants import databases_path


path = databases_path + "/PAMAP2/pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"



list_of_files = [path + '/Protocol/subject101.dat',
                 path +'/Protocol/subject102.dat',
                 path +'/Protocol/subject103.dat',
                 path +'/Protocol/subject104.dat',
                 path +'/Protocol/subject105.dat',
                 path +'/Protocol/subject106.dat',
                 path +'/Protocol/subject107.dat',
                 path +'/Protocol/subject108.dat',
                 path +'/Protocol/subject109.dat' ]

subjectID = [1,2,3,4,5,6,7,8,9]

activityIDdict = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

colNames = ["timestamp", "activityID","heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']



def dataCleaning(dataCollection):
        dataCollection = dataCollection.drop(['handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
                                             'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
                                             'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
                                             axis = 1)  # removal of orientation columns as they are not needed
        dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
        dataCollection = dataCollection.apply(pd.to_numeric, errors='coerce')  # remove non-numeric data in cells
        dataCollection = dataCollection.interpolate() #removal of any remaining NaN value cells by constructing new data points in known set of data points
        
        return dataCollection


def load_pamap2():
    print('Loading pamap2 database...')
    columns = colNames + IMUhand + IMUchest + IMUankle  #all columns in one list
    dataCollection = pd.DataFrame()
    for file in list_of_files:
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = columns
        procData['subject_id'] = int(file[-5])
        dataCollection = pd.concat([dataCollection, pd.DataFrame(procData)], ignore_index=True)

    dataCollection.reset_index(drop=True, inplace=True)
    dataCollection.head()
    dataCol = dataCleaning(dataCollection)
    dataCol.reset_index(drop = True, inplace = True)

    for i in range(0,4):
        dataCol["heartrate"].iloc[i]=100


    pamap2_signals = []
    pamap2_activities = []
    pamap2_activities_indexes = []
    for i in subjectID:
        acc_values_x = dataCol[dataCol['subject_id'] == i]['ankleAcc6_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['ankleAcc6_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['ankleAcc6_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        acc_values_x = dataCol[dataCol['subject_id'] == i]['ankleAcc16_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['ankleAcc16_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['ankleAcc16_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        
        acc_values_x = dataCol[dataCol['subject_id'] == i]['handAcc6_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['handAcc6_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['handAcc6_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        acc_values_x = dataCol[dataCol['subject_id'] == i]['handAcc16_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['handAcc16_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['handAcc16_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        acc_values_x = dataCol[dataCol['subject_id'] == i]['chestAcc6_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['chestAcc6_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['chestAcc6_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        acc_values_x = dataCol[dataCol['subject_id'] == i]['chestAcc16_1'].tolist()
        acc_values_y = dataCol[dataCol['subject_id'] == i]['chestAcc16_2'].tolist()
        acc_values_z = dataCol[dataCol['subject_id'] == i]['chestAcc16_3'].tolist()
        signal = np.array([acc_values_x, acc_values_y, acc_values_z], dtype=float)
        pamap2_signals.append(signal.T)

        activities = dataCol[dataCol['subject_id'] == i]['activityID'].tolist()

        # Find indices where the activity changes
        change_indices = [i for i in range(2, len(activities)) if activities[i] != activities[i-1]] + [len(acc_values_z)]
        
        for a in range(6):
            pamap2_activities.append([activities[0]] + [activities[i] for i in range(2, len(activities)) if activities[i] != activities[i-1]])
            pamap2_activities_indexes.append(change_indices)
            
    print('Loading completed')
    print('Nb_signals : ' + str(len(pamap2_signals)))

    
    return pamap2_signals, pamap2_activities_indexes, pamap2_activities