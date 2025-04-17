# This script explores the 'trials' data in the NWB file,
# plotting the response time ('rt') against the delay for each trial.
# Correcting the delay to be the 'gocuettl' column.

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

# Extract trials data
trials = nwb.intervals["trials"].to_dataframe()
rt = trials['rt']
delay = trials['gocuettl']

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(delay, rt, alpha=0.5)
plt.xlabel('Go Cue Time (s)')
plt.ylabel('Response Time (s)')
plt.title('Response Time vs. Go Cue Time for Each Trial')
plt.grid(True)

# Save the plot to a file
plt.savefig('explore/rt_vs_delay.png')
plt.close()