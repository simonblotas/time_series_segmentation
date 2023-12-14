from wisdm_data_extraction import load_wisdm
from hasctooldataprj_data_extraction import load_hasc
from shoiab_extraction import load_shoiab
from shoiab2_extraction import load_shoiab2
from pamap2_extraction import load_pamap2
from data_pre_processing import segmentations_pre_process, signals_pre_process, signals_to_spectrograms
import numpy as np
import numpy as np
from metrics import crossvalidate, crossvalidate_class
import json
from pipelines import pipelines
from utils import convert_actvivities_to_labels
import jax
jax.config.update('jax_platform_name', 'gpu')
raw_wisdm_signals, raw_wisdm_activities_indexes, raw_wisdm_activities = load_wisdm()
raw_hasc_signals, raw_hasc_activities_indexes, raw_hasc_activities = load_hasc()
raw_shoiab_signals, raw_shoiab_activities_indexes, raw_shoiab_activities = load_shoiab()
raw_shoiab2_signals, raw_shoiab2_activities_indexes, raw_shoiab2_activities = load_shoiab2()
raw_pamap2_signals, raw_pamap2_activities_indexes, raw_pamap2_activities = load_pamap2()

def crop_signals(signals, activities, activities_indexes, signal_max_length):
    # Initialize counters
    count = 0  # Counter for total signals kept
    signals_excluded_count = 0  # Counter for signals excluded due to length

    # Initialize new lists to store cropped signals, activities, and activity indexes
    new_signals, new_activities, new_activities_indexes = [], [], []

    # Iterate through each signal in the dataset
    for i in range(len(signals)):
        # Check if the length of the signal is greater than the specified maximum length
        if len(signals[i]) > signal_max_length:
            # Calculate the number of divisions based on signal_max_length
            n_divs = signals[i].shape[0] // signal_max_length
            
            # Iterate through each division
            buf = 0
            for j in range(n_divs):
                seg = []
                # Extract elements from activities_indexes within the current segment
                for element in activities_indexes[i]:
                    if (
                        element > j * signal_max_length
                        and element < (j + 1) * signal_max_length
                        and element != 0
                    ):
                        seg.append(element % signal_max_length)
                
                # If no elements were found, append signal_max_length to the segment
                if len(seg) == 0:
                    seg.append(signal_max_length)
                
                # If the last element in the segment is not signal_max_length, append it
                if seg[len(seg) - 1] != signal_max_length:
                    seg.append(signal_max_length)
                
                # If the segment has more than one element, update the new lists
                if len(seg) > 1:
                    new_activities_indexes.append(seg)
                    count += 1
                    new_signals.append(signals[i][j * signal_max_length : (j + 1) * signal_max_length])
                    new_activities.append(activities[i][buf:buf+len(new_activities_indexes[-1])])
                    buf += len(new_activities_indexes[-1]) - 1
        else:
            # Increment the signals_excluded_count for signals that don't meet the length criterion
            signals_excluded_count += 1

    # Print summary information
    print('Total signals:', count)
    print('Signals kept:', len(signals) - signals_excluded_count)
    print('Signals excluded:', signals_excluded_count)

    # Return the new lists
    return new_signals, new_activities, new_activities_indexes

# Set the signal_max_length
signal_max_length = 10000

# Set Fourier paramters:
nperseg=300
n_dims = int(nperseg/2 +1)
noverlap=292

# Apply the crop_signals function to multiple datasets
pamap2_signals, pamap2_activities, pamap2_activities_indexes = crop_signals(raw_pamap2_signals, raw_pamap2_activities, raw_pamap2_activities_indexes, signal_max_length)
wisdm_signals, wisdm_activities, wisdm_activities_indexes = crop_signals(raw_wisdm_signals, raw_wisdm_activities, raw_wisdm_activities_indexes, signal_max_length)
hasc_signals, hasc_activities, hasc_activities_indexes = crop_signals(raw_hasc_signals, raw_hasc_activities, raw_hasc_activities_indexes, signal_max_length)
shoiab_signals, shoiab_activities, shoiab_activities_indexes = crop_signals(raw_shoiab_signals, raw_shoiab_activities, raw_shoiab_activities_indexes, signal_max_length)
shoiab2_signals, shoiab2_activities, shoiab2_activities_indexes = crop_signals(raw_shoiab2_signals, raw_shoiab2_activities, raw_shoiab2_activities_indexes, signal_max_length)

# Apply the pre_process function to multiple datasets
wisdm_pre_process_sig = signals_pre_process(wisdm_signals, sample_rate = 20., cutoff_freq=6)
pamap2_pre_process_sig = signals_pre_process(pamap2_signals, sample_rate = 100., cutoff_freq=6)
hasc_pre_process_sig = signals_pre_process(hasc_signals, sample_rate = 100., cutoff_freq=6)
shoiab_pre_process_sig = signals_pre_process(shoiab_signals, sample_rate = 50., cutoff_freq=6)
shoiab2_pre_process_sig = signals_pre_process(shoiab2_signals, sample_rate = 50., cutoff_freq=6)

# Apply the pre_process function to multiple datasets
wisdm_spectrograms = signals_to_spectrograms(wisdm_pre_process_sig, nperseg=nperseg, noverlap=noverlap, fs=20)
hasc_spectrograms = signals_to_spectrograms(hasc_pre_process_sig, nperseg=nperseg, noverlap=noverlap, fs=100)
pamap2_spectrograms = signals_to_spectrograms(pamap2_pre_process_sig, nperseg=nperseg, noverlap=noverlap, fs=100)
shoiab_spectrograms = signals_to_spectrograms(shoiab_pre_process_sig, nperseg=nperseg, noverlap=noverlap, fs=50)
shoiab2_spectrograms = signals_to_spectrograms(shoiab2_pre_process_sig, nperseg=nperseg, noverlap=noverlap, fs=50)


wisdm_label_dict = {'Jogging': 0, "Walking": 1, 'Upstairs': 2,'Downstairs': 3, 'Sitting': 4, 'Standing' : 5}
hasc_label_dict = {'walk': 0, "jog": 1, 'stay': 2,'stDown': 3, 'stUp': 4, 'skip' : 5}
pamap2_label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5 : 4, 6 : 5, 7:6, 8:7, 17:8, 16:9, 12:10, 13:11, 24:12}
shoiab_label_dict = {'walking': 0, "standing": 1, 'jogging': 2,'sitting': 3, 'biking': 4, 'upstairs' : 5, 'upsatirs':5, 'downstairs':6 }
shoiab2_label_dict = {'Downstairs': 0, "Running": 1, 'Sitting': 2,'Standing': 3, 'Upstairs': 4, 'Walking' : 5, 'Downstairs':6 }


# Define the parameters of folds for cross-validation
num_folds = 3
num_epochs = 100
batch_size = 5
marges = [5/100, 4/100, 3/100, 2/100, 1/100]

# List of database names
#databases_names = ["wisdm", "hasc", "pamap2", "shoiab", "shoiab2"]
databases_names = [ "shoiab2"]
#pipelines_names = ["trans_seg","trans_bin", "trans_seg_fourier", "trans_bin_fourier", "trans_seg_class", "trans_seg_class_fourier", "trans_bin_class","trans_bin_class_fourier"]
pipelines_names = ["trans_seg_class_fourier"]
# Entry dictionary
entry = {
    "wisdm": {'signals': wisdm_pre_process_sig,'fourier_signals': wisdm_spectrograms,'segmentations': wisdm_activities_indexes, "activities": wisdm_activities,"nb_class" : 6, "dict_class" : wisdm_label_dict},
    "hasc": {'signals': hasc_pre_process_sig, 'fourier_signals': hasc_spectrograms, 'segmentations': hasc_activities_indexes, "activities": hasc_activities, "nb_class" : 6, "dict_class" : hasc_label_dict},
    "pamap2": {'signals': pamap2_pre_process_sig, 'fourier_signals': pamap2_spectrograms,'segmentations': pamap2_activities_indexes, "activities": pamap2_activities, "nb_class" : 13, "dict_class" : pamap2_label_dict},  
    "shoiab": {'signals': shoiab_pre_process_sig, 'fourier_signals': shoiab_spectrograms,'segmentations': shoiab_activities_indexes, "activities": shoiab_activities, "nb_class" : 7, "dict_class" : shoiab_label_dict},  
    "shoiab2": {'signals': shoiab2_pre_process_sig, 'fourier_signals': shoiab2_spectrograms,'segmentations': shoiab2_activities_indexes, "activities": shoiab2_activities, "nb_class" : 7, "dict_class" : shoiab2_label_dict} 
}




for pipeline_name in pipelines_names:
    pipeline = pipelines[pipeline_name]
    model = pipeline['pipeline']
    transformed_signal_length = int(pipeline['transformed_signal_length'] * signal_max_length)
    
    if pipeline['class'] == False : 
        for database_name in databases_names:

            if pipeline['fourier']:
                X = entry[database_name]['fourier_signals']
            else:
                X = entry[database_name]['signals']
            _ , _, y = segmentations_pre_process(entry[database_name]['segmentations'],transformed_signal_length, signal_max_length)
            print("Test on: ", pipeline_name,database_name)
            test_scores = crossvalidate(model, X, y, num_folds, num_epochs, batch_size, verbose= True, marges = marges)

            file_result = "/home/sblotas/segmentation_notebook_version_v2/results_benchmark/{}_{}.json".format(pipeline_name,database_name)

            mean_test_score = np.mean(test_scores, axis = 1)
            std_test_score = np.std(test_scores, axis = 1)
            # Create a dictionary to store the data
            data_dict = {
                "test_scores_mean": list(mean_test_score),
                "test_scores_std": list(std_test_score),
                "test_scores_marges_5%": list(test_scores[0]),
                "test_scores_marges_4%": list(test_scores[1]),
                "test_scores_marges_3%": list(test_scores[2]),
                "test_scores_marges_2%": list(test_scores[3]),
                "test_scores_marges_1%": list(test_scores[4]),
            }

            # Write the data to the JSON file
            with open(file_result, "w") as json_file:
                json.dump(data_dict, json_file, indent=4)
    else:
        
        for database_name in databases_names:
            model_class = model[entry[database_name]['nb_class']]
            
            if pipeline['fourier']:
                X = entry[database_name]['fourier_signals']
            else:
                X = entry[database_name]['signals']
            _ , size_adapted_seg, y = segmentations_pre_process(entry[database_name]['segmentations'],transformed_signal_length, signal_max_length)
            true_classification_size_adapted = convert_actvivities_to_labels(entry[database_name]['activities'], size_adapted_seg, entry[database_name]['dict_class'], entry[database_name]['nb_class'])
            print("Test on: ", pipeline_name,database_name)
            test_scores = crossvalidate_class(model_class, X, y,true_classification_size_adapted, num_folds, num_epochs, batch_size, verbose= True, marges = marges)

            file_result = "/home/sblotas/time_series_segmentation/results_benchmark/{}_{}.json".format(pipeline_name,database_name)

            mean_test_score = np.mean(test_scores, axis = 1)
            std_test_score = np.std(test_scores, axis = 1)
            # Create a dictionary to store the data
            data_dict = {
                "test_scores_mean": list(mean_test_score),
                "test_scores_std": list(std_test_score),
                "test_scores_marges_5%": list(test_scores[0]),
                "test_scores_marges_4%": list(test_scores[1]),
                "test_scores_marges_3%": list(test_scores[2]),
                "test_scores_marges_2%": list(test_scores[3]),
                "test_scores_marges_1%": list(test_scores[4]),
            }

            # Write the data to the JSON file
            with open(file_result, "w") as json_file:
                json.dump(data_dict, json_file, indent=4)