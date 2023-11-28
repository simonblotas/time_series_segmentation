import os
# Define the parent folder
parent_folder = "Databases"

# Create the parent folder
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
    print(os.makedirs(parent_folder))

WISDM_MIN_LENGHT = 5000
databases_path = None # ADAPT TO YOUR MACHINE, example : '/time_series_segmentation/Databases'

# Check if the database_folder_path is not specified (empty or None)
if not databases_path:
    raise ValueError("Database folder path not specified. Please set the 'databases_path' variable in main_constant.py.")