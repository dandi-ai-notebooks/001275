# This script loads an NWB file remotely and plots the first ~50,000 samples of:
# 1. Eye position (x and y) over time
# 2. Hand position (voltage) over time
# The purpose is to get an initial overview of the behavioral signals during the task.
# Generated images saved as PNGs in the tmp_scripts directory.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

# Access eye position
eye_series = nwb.processing["behavior"].data_interfaces["eye_position"]
eye_data = eye_series.data[:50000, :]  # load first 50k samples (x,y)
eye_times = eye_series.timestamps[:50000]

plt.figure(figsize=(10,5))
plt.plot(eye_times, eye_data[:,0], label="Eye X")
plt.plot(eye_times, eye_data[:,1], label="Eye Y")
plt.xlabel("Time (s)")
plt.ylabel("Position (meters)")
plt.title("Initial 50,000 samples of Eye Position")
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/eye_position.png")
plt.close()

# Access hand position
hand_series = nwb.processing["behavior"].data_interfaces["hand_position"]
hand_data = hand_series.data[:50000]
hand_times = hand_series.timestamps[:50000]

plt.figure(figsize=(10,5))
plt.plot(hand_times, hand_data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.title("Initial 50,000 samples of Hand Position")
plt.tight_layout()
plt.savefig("tmp_scripts/hand_position.png")
plt.close()