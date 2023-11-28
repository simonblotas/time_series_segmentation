import numpy as np
from main_constants import WISDM_MIN_LENGHT
from main_constants import databases_path


path = databases_path + "/WISDM/WISDM_ar_v1.1_raw.txt"

min_lenght = WISDM_MIN_LENGHT

signal_lenght = 1

def only_stand_or_sit(list_activity):
    for element in list_activity:
        if element != "Standing" and element != "Sitting":
            return False
    return True


def stairs(list_activity):
    for element in list_activity:
        if element == "Upstairs" or element == "Downstairs":
            return True
    return False


def load_wisdm():
    print("Loading WISDM database ...")
    with open(path, "r") as file:
        lines = file.read().split(";")

    # Initialize variables to store data
    current_user_id = None
    current_user_data = []

    # Initialize a dictionary to store data for each user
    user_data = {}
    count_signals = 0
    previous_timestamp = 0
    # Process each line
    for line in lines:
        if line.strip():  # Skip empty lines
            # Split the line into fields using commas as the delimiter
            fields = line.strip().split(",")

            if len(fields) == 6:  # Ensure correct number of fields
                # Extract data from the fields
                user_id = fields[0]
                activity = fields[1]
                timestamp = int(fields[2])
                x_acceleration = float(fields[3])
                y_acceleration = float(fields[4])
                z_acceleration = float(fields[5])

                # Check if the user_id has changed
                if user_id != current_user_id or (
                    timestamp != 0
                    and previous_timestamp != 0
                    and np.abs(timestamp - previous_timestamp) > 1e13
                ):
                    # Save the data for the previous user, if any
                    if current_user_id is not None:
                        user_data[count_signals] = current_user_data

                    # Initialize data for the new user
                    current_user_id = user_id
                    current_user_data = []
                    count_signals += 1

                # Create a data point dictionary for the current record
                data_point = {
                    "activity": activity,
                    "timestamp": timestamp,
                    "x_acceleration": x_acceleration,
                    "y_acceleration": y_acceleration,
                    "z_acceleration": z_acceleration,
                }

                # Append the data point to the current user's data list
                current_user_data.append(data_point)
                previous_timestamp = timestamp

    # Save the data for the last user
    if current_user_id is not None:
        user_data[count_signals] = current_user_data

    # Now user_data is a dictionary containing lists of data points for each user
    # You can access the data for a specific user using user_data[user_id]

    # Extract all user IDs from the dataset
    user_ids = list(user_data.keys())

    # Create a dictionary to store signal data arrays and activity changes for each user
    user_data_info = {}

    # Iterate through each user's data
    for user_id, user_signal_data in user_data.items():
        # Create NumPy array for acceleration data
        acceleration_data = np.array(
            [
                [
                    data_point["x_acceleration"],
                    data_point["y_acceleration"],
                    data_point["z_acceleration"],
                ]
                for data_point in user_signal_data
            ]
        )

        # Create a list to track activity changes and their indexes
        activity_changes = [user_signal_data[0]["activity"]]
        activity_change_indexes = []
        for i in range(1, len(user_signal_data)):
            if user_signal_data[i]["activity"] != user_signal_data[i - 1]["activity"]:
                activity_changes.append(user_signal_data[i]["activity"])
                activity_change_indexes.append(i)
        activity_change_indexes.append(len(user_signal_data))

        # Convert the list to a NumPy array
        activity_change_indexes_array = np.array(activity_change_indexes)

        # Store the acceleration data array and activity change indexes array in the dictionary
        user_data_info[user_id] = {
            "acceleration_data": acceleration_data,
            "activity_changes": activity_changes,
            "activity_change_indexes": activity_change_indexes_array,
        }

    # Now user_data_info contains information about signal data, activity changes, and indexes for each user
    # You can access the information for a specific user using user_data_info[user_id]

    wisdm_signals = []
    wisdm_segmentations = []
    wisdm_activities = []




    for user_id in user_ids:
        
        if (
            len(user_data_info[user_id]["activity_changes"]) > 0
            and user_data_info[user_id]["acceleration_data"].shape[0] > min_lenght
            and only_stand_or_sit(user_data_info[user_id]["activity_changes"]) == False
            and stairs(user_data_info[user_id]["activity_changes"])
        ):
            wisdm_signals.append(
                    user_data_info[user_id]["acceleration_data"]
                    )
            seg = [i for i in user_data_info[user_id]["activity_change_indexes"]]

            wisdm_segmentations.append(seg)
            wisdm_activities.append(user_data_info[user_id]["activity_changes"][:len(seg)])

            if user_data_info[user_id]["acceleration_data"].shape[0] > signal_lenght:
                n_divs = (
                    user_data_info[user_id]["acceleration_data"].shape[0]
                    // signal_lenght
                )
                
                for i in range(n_divs):
                    seg = []
                    for element in user_data_info[user_id]["activity_change_indexes"]:
                        if (
                            element > i * signal_lenght
                            and element < (i + 1) * signal_lenght
                            and element != 0
                        ):
                            seg.append(element % signal_lenght)
                    if len(seg) == 0:
                        seg.append(signal_lenght)
                    if seg[len(seg) - 1] != signal_lenght:
                        seg.append(signal_lenght)
                    if len(seg) > 1:
                        pass
                        #wisdm_signals.append(
                        #    user_data_info[user_id]["acceleration_data"][
                        #       i * signal_lenght : (i + 1) * signal_lenght
                        #    ]
                        #)
                        
                        #wisdm_segmentations.append(seg)
                        #wisdm_activities.append(user_data_info[user_id]["activity_changes"][:len(seg)])
                        

            if user_data_info[user_id]["acceleration_data"].shape[0] < signal_lenght:
                num_repeats = (
                    signal_lenght
                    // user_data_info[user_id]["acceleration_data"].shape[0]
                    + 1
                )
                expanded_signal = np.tile(
                    user_data_info[user_id]["acceleration_data"], (num_repeats, 1)
                )

                seg = []
                n_lenght = user_data_info[user_id]["acceleration_data"].shape[0]
                for j in range(num_repeats):
                    for i in range(
                        len(user_data_info[user_id]["activity_change_indexes"])
                    ):
                        indx = (
                            user_data_info[user_id]["activity_change_indexes"][i]
                            % n_lenght
                            + j * n_lenght
                        )
                        if 0 < indx < signal_lenght:
                            seg.append(indx)
                if len(seg) == 0:
                    seg.append(signal_lenght)
                if seg[len(seg) - 1] != signal_lenght:
                    seg.append(signal_lenght)
                if len(seg) > 1:
                    #wisdm_signals.append(expanded_signal[:signal_lenght])
                    #wisdm_segmentations.append(seg)
                    #wisdm_activities.append(user_data_info[user_id]["activity_changes"][:len(seg)])
                    pass
    print("Loading completed")
    print("Nb_signals : ", len(wisdm_signals))
    return wisdm_signals, wisdm_segmentations, wisdm_activities
