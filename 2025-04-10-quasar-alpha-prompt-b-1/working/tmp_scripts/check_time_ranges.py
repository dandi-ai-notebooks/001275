#!/usr/bin/env python3
"""
Script to check the min and max timestamps for eye and hand position data
in the chosen NWB file from Dandiset 001275.

Goal: identify a good snippet window with actual data recording.
"""

import remfile
import h5py
import pynwb
import numpy as np

url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

eye_pos = nwb.processing["behavior"].data_interfaces["eye_position"]
hand_pos = nwb.processing["behavior"].data_interfaces["hand_position"]

eye_times = eye_pos.timestamps[:]
hand_times = hand_pos.timestamps[:]

print(f"Eye position timestamps: min={np.min(eye_times):.2f} s, max={np.max(eye_times):.2f} s")
print(f"Hand position timestamps: min={np.min(hand_times):.2f} s, max={np.max(hand_times):.2f} s")