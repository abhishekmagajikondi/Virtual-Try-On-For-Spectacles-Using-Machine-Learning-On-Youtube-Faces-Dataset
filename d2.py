import os
import shutil
import pandas as pd

# Path to the CSV file
csv_file_path = 'valid_samples.csv'

# Path to the training folder
training_folder_path = "training"

# Path to the validation folder
validation_folder_path = "validating"

# Load CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Get the image name from the 0th column
    image_name = row.iloc[0]

    # Build the paths for the source and destination images
    source_path = os.path.join(training_folder_path, image_name)
    destination_path = os.path.join(validation_folder_path, image_name)

    # Move the image from the training folder to the validation folder
    shutil.move(source_path, destination_path)

print("Images moved successfully!")
