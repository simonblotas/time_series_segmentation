WISDM_MIN_LENGHT = 5000
databases_path = None # ADAPT TO YOUR MACHINE, example : '/time_series_segmentation/Databases'

# Check if the database_folder_path is not specified (empty or None)
if not databases_path:
    raise ValueError("Database folder path not specified. Please set the 'databases_path' variable in main_constant.py.")
