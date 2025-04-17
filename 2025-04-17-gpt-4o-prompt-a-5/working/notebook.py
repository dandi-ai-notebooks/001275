# %% [markdown]
# # Exploring Dandiset 001275: Mental Navigation in Primates
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# - **Dataset Name**: Mental navigation primate PPC Neupane_Fiete_Jazayeri
# - **Dataset Description**: Neurophysiology data collected from two primates during a mental navigation task with data from the entorhinal cortex.
# - **Neurosift Link**: [https://neurosift.app/dandiset/001275](https://neurosift.app/dandiset/001275)
# 
# This dataset contains data relevant for studying brain function and organization during a mental navigation task.

# %% [markdown]
# ## Notebook Outline
# - Load the dataset using DANDI API
# - View metadata of the selected NWB file
# - Visualize selected data from the NWB file

# %% [markdown]
# ## Required Packages
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `matplotlib` (for visualization)

# %% [markdown]
# ## Loading the Dandiset
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading the NWB File
import pynwb
import h5py
import remfile

# Specify the file URL
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Display metadata
print(nwb.session_description)
print(nwb.identifier)
print(nwb.session_id)
print(nwb.session_start_time)

# %% [markdown]
# ## Data Visualization
import matplotlib.pyplot as plt

# Access eye position data
eye_position = nwb.processing["behavior"].data_interfaces["eye_position"]

# Plot eye position over time
plt.figure(figsize=(10, 4))
plt.plot(eye_position.timestamps[0:1000], eye_position.data[0:1000, 0], label="X position")
plt.plot(eye_position.timestamps[0:1000], eye_position.data[0:1000, 1], label="Y position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Eye Position Over Time")
plt.legend()
plt.show()

# %% [markdown]
# ## Summary
# This notebook provided an overview of the neurophysiology data in Dandiset 001275, demonstrated how to load data using the DANDI API, and visualized eye position data over time. Future analyses could explore more detailed behavioral patterns or electrophysiological data.