import pandas as pd
import os


def load_and_combine_pkl_files(directory_path):
    # List to hold all the dataframes
    dataframes = []

    # Loop through all the files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)
            # Load the dataframe from a pkl file
            df = pd.read_pickle(file_path)
            # Append the dataframe to the list
            dataframes.append(df)

    # Concatenate all dataframes into one big dataframe
    big_df = pd.concat(dataframes, ignore_index=True)

    return big_df


# Usage
directory_path = '/Users/ximing/Desktop/Explainprompt/similarity/'
big_df = load_and_combine_pkl_files(directory_path)
print(big_df)
