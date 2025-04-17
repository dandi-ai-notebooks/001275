# This script explores the 'eye_position' data in the NWB file,
# plotting the x and y eye positions over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract eye position data
eye_position = nwb.processing["behavior"].data_interfaces["eye_position"]
eye_position_x = eye_position.data[:, 0]
eye_position_y = eye_position.data[:, 1]
timestamps = eye_position.timestamps[:]

# Take the first 10000 samples to prevent the notebook from timing out
num_samples = 10000
eye_position_x = eye_position_x[:num_samples]
eye_position_y = eye_position_y[:num_samples]
timestamps = timestamps[:num_samples]

# Subtract the minimum timestamp from all timestamps
timestamps = timestamps - np.min(timestamps)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(timestamps, eye_position_x, label='Eye Position X')
plt.plot(timestamps, eye_position_y, label='Eye Position Y')
plt.xlabel('Time (s)')
plt.ylabel('Eye Position (meters)')
plt.title('Eye Position over Time')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('explore/eye_position_over_time.png')
plt.close()