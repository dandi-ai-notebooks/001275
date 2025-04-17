# %% [markdown]
# Exploring Dandiset 001275: Mental Navigation in Primates

# %% [markdown]
# **Important:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of Dandiset 001275
#
# This dataset contains neurophysiology data collected from two primates during a mental navigation task associated with a previously published study (https://doi.org/10.1038/s41586-024-07557-z).
#
# - [Neurosift Link](https://neurosift.app/dandiset/001275)
#
# This notebook will guide you through loading and visualizing data from this Dandiset.

# %% [markdown]
# ## What this notebook will cover:
#
# 1.  Loading the Dandiset using the DANDI API.
# 2.  Loading an NWB file from the Dandiset.
# 3.  Exploring and visualizing data from the NWB file, including eye position and hand position data.
# 4.  Examining trial data.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook. Make sure they are installed in your environment.
#
# -   pynwb
# -   h5py
# -   remfile
# -   matplotlib
# -   numpy
# -   seaborn

# %%
# Load required packages
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% [markdown]
# ## Loading the Dandiset
#
# Use the DANDI API to connect to the Dandiset.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading an NWB File
#
# We will load the file `sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb` to demonstrate how to access data.
# This file was chosen because it's relatively small.
# The URL for this asset is: https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb

# %% [markdown]
# ## Exploring NWB File Metadata
#
# Let's explore some of the metadata in the NWB file.

# %%
nwb.session_description  # (str) Data from macaque performing mental navigation task. Subject is presented with a start and a targ...

# %%
nwb.identifier  # (str) 15de6847-1f57-4247-847b-af4b179d1b7c

# %%
nwb.session_start_time  # (datetime) 2020-01-04T00:00:00-05:00

# %% [markdown]
# ## Exploring Behavior Data (Eye Position)
#
# Let's load and visualize some eye position data.

# %%
eye_position_data = nwb.processing["behavior"].data_interfaces["eye_position"].data
eye_position_timestamps = nwb.processing["behavior"].data_interfaces["eye_position"].timestamps

# %%
# Plot the first 10 seconds of eye position data
duration = 10  # seconds
sample_rate = 100  # samples per second
num_samples = duration * sample_rate
eye_position_data_subset = eye_position_data[:num_samples, :]
eye_position_timestamps_subset = eye_position_timestamps[:num_samples]

plt.figure(figsize=(10, 5))
plt.plot(eye_position_timestamps_subset, eye_position_data_subset[:, 0], label="Eye Position X")
plt.plot(eye_position_timestamps_subset, eye_position_data_subset[:, 1], label="Eye Position Y")
plt.xlabel("Time (s)")
plt.ylabel("Eye Position (meters)")
plt.title("Eye Position Data")
plt.legend()
plt.show()

# %% [markdown]
# The above plot shows the eye position data for the first 10 seconds of the recording. You can see the X and Y coordinates of the eye position over time.

# %% [markdown]
# ## Exploring Behavior Data (Hand Position)
#
# Let's load and visualize some hand position data.

# %%
hand_position_data = nwb.processing["behavior"].data_interfaces["hand_position"].data
hand_position_timestamps = nwb.processing["behavior"].data_interfaces["hand_position"].timestamps

# %%
# Plot the first 10 seconds of hand position data
duration = 10  # seconds
sample_rate = 100  # samples per second
num_samples = duration * sample_rate
hand_position_data_subset = hand_position_data[:num_samples]
hand_position_timestamps_subset = hand_position_timestamps[:num_samples]

plt.figure(figsize=(10, 5))
plt.plot(hand_position_timestamps_subset, hand_position_data_subset, label="Hand Position")
plt.xlabel("Time (s)")
plt.ylabel("Hand Position (voltage)")
plt.title("Hand Position Data")
plt.legend()
plt.show()

# %% [markdown]
# The above plot shows the hand position data for the first 10 seconds of the recording.

# %% [markdown]
# ## Exploring Trial Data
#
# Let's view data from the `trials` table.

# %%
nwb.trials.to_dataframe().head()

# %% [markdown]
# We are displaying the first 5 rows of the trials table, including start time, stop time, and other trial-related data.

# %% [markdown]
# ## Exploring Units Data
#
# Now let's explore the ecephys data, starting with the units table.

# %%
units = nwb.processing["ecephys"].data_interfaces["units"]
units.to_dataframe().head()

# %% [markdown]
#

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to load and visualize data from Dandiset 001275 using the DANDI API and PyNWB. We explored eye position, hand position, and trial data.
#
# Possible future directions for analysis include:
#
# -   Analyzing the relationship between eye position and hand position during the mental navigation task.
# -   Investigating the neural activity recorded in the ecephys data.
# -   Performing more detailed analysis of the trial data to understand the task performance.