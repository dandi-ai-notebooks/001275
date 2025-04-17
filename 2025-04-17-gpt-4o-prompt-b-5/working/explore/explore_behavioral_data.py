# Aim: Explore the behavioral data (eye and hand positions) from the NWB file.

import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile

# Load the NWB file remotely
url = "https://api.dandiarchive.org/api/assets/d07034d5-a822-4247-bbd1-97f67921a1d3/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract eye position data (first 10,000 samples for manageable visualization)
eye_position_data = nwb.processing["behavior"].data_interfaces["eye_position"].data[:10000, :]
eye_position_timestamps = nwb.processing["behavior"].data_interfaces["eye_position"].timestamps[:10000]

# Plot Eye Position Data
plt.figure(figsize=(12, 6))
plt.plot(eye_position_timestamps, eye_position_data[:, 0], label='Horizontal Position')
plt.plot(eye_position_timestamps, eye_position_data[:, 1], label='Vertical Position', alpha=0.7)
plt.title('Eye Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (meters)')
plt.legend()
plt.grid(True)
plt.savefig('explore/eye_position_plot.png')
plt.close()

# Extract hand position data (first 10,000 samples for manageable visualization)
hand_position_data = nwb.processing["behavior"].data_interfaces["hand_position"].data[:10000]
hand_position_timestamps = nwb.processing["behavior"].data_interfaces["hand_position"].timestamps[:10000]

# Plot Hand Position Data
plt.figure(figsize=(12, 6))
plt.plot(hand_position_timestamps, hand_position_data, label='Hand Position')
plt.title('Hand Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (voltage)')
plt.legend()
plt.grid(True)
plt.savefig('explore/hand_position_plot.png')
plt.close()