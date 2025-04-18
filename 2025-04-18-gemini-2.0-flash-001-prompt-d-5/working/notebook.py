# %% [markdown]
# # Exploring Dandiset 001275: Mental Navigation in Primates
#
# **Important:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Exercise caution when interpreting the code or results.
#
# ## Overview of the Dandiset
#
# This Dandiset (001275) contains neurophysiology data collected from two primates during a mental navigation task.
# The data is associated with a previously published study (https://doi.org/10.1038/s41586-024-07557-z).
# Data from the entorhinal cortex is open-sourced here: https://doi.org/10.48324/dandi.000897/0.240605.1710
#
# Here is a neurosift link for the dandiset: https://neurosift.app/dandiset/001275
#
# ## What this notebook will cover
#
# In this notebook, we will load the Dandiset using the DANDI API, explore the available assets,
# load data from an example NWB file, and visualize some of the data.
#
# ## Required Packages
#
# The following packages are required to run this notebook:
#
# - pynwb
# - h5py
# - remfile
# - numpy
# - matplotlib
# - seaborn
# - pandas
#
# These packages are assumed to be already installed in your environment.
#
# %%
# Load Dandiset using the DANDI API
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
# ## Load and explore an NWB file
#
# We will now load one of the NWB files in the Dandiset and explore some of its metadata and data.
# We will load the file `sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb`.
#
# The URL for this asset is: https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/

# %%
# Load the NWB file
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

print(nwb)
print(nwb.session_description)
print(nwb.identifier)
print(nwb.session_start_time)

# %% [markdown]
# ## Load and visualize eye position data
# Now we will load and visualize the eye position data from the NWB file.

# %%
# Load a subset of eye position data and timestamps
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
eye_position_data = nwb.processing["behavior"].data_interfaces["eye_position"].data[:10000, :]
eye_position_timestamps = nwb.processing["behavior"].data_interfaces["eye_position"].timestamps[:10000]

# Normalize timestamps
eye_position_timestamps = eye_position_timestamps - eye_position_timestamps[0]

# Plot eye position over time
plt.figure(figsize=(10, 6))
plt.plot(eye_position_timestamps, eye_position_data[:, 0], label="Eye position X")
plt.plot(eye_position_timestamps, eye_position_data[:, 1], label="Eye position Y")
plt.xlabel("Time (s)")
plt.ylabel("Eye position (pixels)")
plt.title("Eye position over time")
plt.legend()
plt.show()

# %% [markdown]
# ## Load and visualize unit data (spike counts)
# Now we will load and visualize the distribution of spike counts across units.

# %%
# Access the units data
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme()
units = nwb.processing["ecephys"].data_interfaces["units"]

# Convert units data to a DataFrame
units_df = units.to_dataframe()

# Print the first few rows of the DataFrame
print(units_df.head())

# Plotting the distribution of spike counts for each unit
plt.figure(figsize=(10,6))
plt.hist(units_df['n_spikes'], bins=30)
plt.xlabel('Number of Spikes')
plt.ylabel('Number of Units')
plt.title('Distribution of Spike Counts Across Units')
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# In this notebook, we have demonstrated how to load a DANDI dataset using the DANDI API,
# load an NWB file from the dataset, and visualize some example data.
#
# Possible future directions for analysis include:
#
# - Exploring other data modalities in the NWB file, such as LFP data.
# - Performing more detailed analysis of the eye position data, such as identifying saccades.
# - Analyzing the relationship between neural activity and behavior.