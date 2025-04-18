import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()

# Script to load unit data (spike times) from the NWB file
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access the units data
units = nwb.processing["ecephys"].data_interfaces["units"]

# Convert units data to a DataFrame
units_df = units.to_dataframe()

# Print the first few rows of the DataFrame
print(units_df.head())

# Plotting the distribution of spike counts for each unit
plt.figure(figsize=(10,6))
plt.hist(units_df['n_spikes'], bins=30)
plt.xlabel('Number of Spikes')
plt.ylabel('Number of Units')
plt.title('Distribution of Spike Counts Across Units')
plt.savefig('explore/spike_count_distribution.png')
plt.close()

print("Unit data loaded and spike count distribution plotted. See explore/spike_count_distribution.png")