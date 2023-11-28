import os
# Define the parent folder
parent_folder = "Databases"

# Create the parent folder
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
    print(os.makedirs(parent_folder))

WISDM_MIN_LENGHT = 5000
databases_path = '/home/sblotas/segmentation_notebook_version_v2/Databases'