import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Script to load eye position data from the NWB file and plot a subset
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Load a subset of eye position data and timestamps
eye_position_data = nwb.processing["behavior"].data_interfaces["eye_position"].data[:10000, :]
eye_position_timestamps = nwb.processing["behavior"].data_interfaces["eye_position"].timestamps[:10000]

# Plot eye position over time
plt.figure(figsize=(10, 6))
plt.plot(eye_position_timestamps, eye_position_data[:, 0], label="Eye position X")
plt.plot(eye_position_timestamps, eye_position_data[:, 1], label="Eye position Y")
plt.xlabel("Time (s)")
plt.ylabel("Eye position (meters)")
plt.title("Eye position over time")
plt.legend()
plt.savefig("explore/eye_position_over_time.png")
plt.close()

print("Eye position data loaded and plotted.  See explore/eye_position_over_time.png")