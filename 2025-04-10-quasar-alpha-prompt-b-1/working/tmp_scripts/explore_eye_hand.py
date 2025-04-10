#!/usr/bin/env python3
"""
Exploratory script: load small snippet (~5 seconds) of eye & hand position data
from valid timestamp window in Dandiset 001275 NWB file.

Plots saved to tmp_scripts directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import remfile
import h5py
import pynwb

url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

eye_pos = nwb.processing["behavior"].data_interfaces["eye_position"]
hand_pos = nwb.processing["behavior"].data_interfaces["hand_position"]

eye_times = eye_pos.timestamps
hand_times = hand_pos.timestamps

# New time window where behavioral data actually exists
time_start = 704315
time_end = 704320

eye_slice = np.where((eye_times[:] >= time_start) & (eye_times[:] <= time_end))[0]
hand_slice = np.where((hand_times[:] >= time_start) & (hand_times[:] <= time_end))[0]

max_points = 5000
eye_slice = eye_slice[:max_points]
hand_slice = hand_slice[:max_points]

eye_data = eye_pos.data[eye_slice, :]  # shape (N, 2)
eye_timestamps = eye_times[eye_slice]

hand_data = hand_pos.data[hand_slice]
hand_timestamps = hand_times[hand_slice]

plt.figure(figsize=(10, 4))
plt.plot(eye_timestamps, eye_data[:, 0], label="Eye X [m]")
plt.plot(eye_timestamps, eye_data[:, 1], label="Eye Y [m]")
plt.xlabel("Time (s)")
plt.ylabel("Position (meters)")
plt.title("Eye position over 5 s snippet")
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/eye_position_timeseries.png")
plt.close()

plt.figure(figsize=(5, 5))
plt.plot(eye_data[:, 0], eye_data[:, 1], lw=0.5)
plt.xlabel("Eye X (meters)")
plt.ylabel("Eye Y (meters)")
plt.title("2D Eye trajectory snippet")
plt.tight_layout()
plt.savefig("tmp_scripts/eye_position_2d.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(hand_timestamps, hand_data, label="Hand position [V]", color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.title("Hand position over 5 s snippet")
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/hand_position_timeseries.png")
plt.close()