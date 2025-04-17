# This script explores the 'units' data in the NWB file,
# creating a histogram of the spike counts for each unit.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract units data
units = nwb.processing["ecephys"].data_interfaces["units"].to_dataframe()
spike_counts = units['n_spikes']

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(spike_counts, bins=30, alpha=0.7)
plt.xlabel('Number of Spikes')
plt.ylabel('Number of Units')
plt.title('Distribution of Spike Counts for Each Unit')
plt.grid(True)

# Save the plot to a file
plt.savefig('explore/spike_count_histogram.png')
plt.close()