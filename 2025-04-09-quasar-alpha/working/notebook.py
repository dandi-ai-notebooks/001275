# %% [markdown]
# # Mental navigation primate PPC Dandiset 001275 exploration
#
# **Disclaimer:** This notebook was generated automatically using `dandi-notebook-gen`. The code and analyses have **not** been thoroughly reviewed by a human expert. You should critically evaluate the analyses and interpretations before relying on them.
#
# This notebook demonstrates how to explore the Dandiset **Mental navigation primate PPC Neupane_Fiete_Jazayeri (ID: 001275)**, containing neurophysiology recordings during a primate mental navigation task.
#
# The data include multi-electrode neural recordings, behavioral measurements (eye, hand movements), metadata on sessions, and more. The goal is to get started loading, analyzing, and visualizing this rich dataset.

# %% [markdown]
# ## About the dataset
#
# - **Title**: Mental navigation primate PPC Neupane_Fiete_Jazayeri
# - **Dandiset ID**: 001275
# - **Version**: draft (2024-12-05)
# - **Contributors**: Sujaya Neupane
# - **Citation**: Neupane, Sujaya (2024) *Mental navigation primate PPC Neupane_Fiete_Jazayeri (Version draft)* [Data set]. DANDI archive. https://dandiarchive.org/dandiset/001275/draft
# - **Description**: Neurophysiology data collected from two primates during a mental navigation task, including extracellular electrophysiology, behavioral data, and spike sorting results.
# - **Related study**: https://doi.org/10.1038/s41586-024-07557-z
#
# Data open sourced here focuses on recordings from the posterior parietal cortex.

# %% [markdown]
# ## List assets in this dandiset
#
# The cell below connects to DANDI and lists available files.

# %%
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets in this Dandiset.")
for asset in assets:
    print(asset.path)

# %% [markdown]
# ## Loading a sample NWB file remotely
#
# We now load one integrated behavioral + ephys NWB file:
#
# `sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb`
#
# This is done **without fully downloading** the file via `remfile` and `h5py`.

# %%
import pynwb
import h5py
import remfile

nwb_url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
file = remfile.File(nwb_url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

print("NWB Session:", nwb.session_description)
print("Session ID:", nwb.session_id)
print("Subject ID:", nwb.subject.subject_id)
print("Species:", nwb.subject.species)
print("Session Start Time:", nwb.session_start_time)

# %% [markdown]
# ## Behavioral signals overview
#
# Here we explore behavioral data streams (eye and hand movement).
#
# The plots below show initial snippets (~50,000 samples). You can adjust the sample window or access subsets over different intervals similarly.
#
# **Note:** Remote NWB files can be very large. When accessing data, be mindful to **load manageable chunks**.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Eye position
eye_series = nwb.processing["behavior"].data_interfaces["eye_position"]
eye_data = eye_series.data[:50000, :]
eye_times = eye_series.timestamps[:50000]

plt.figure(figsize=(10,5))
plt.plot(eye_times, eye_data[:,0], label='Eye X')
plt.plot(eye_times, eye_data[:,1], label='Eye Y')
plt.xlabel('Time (s)')
plt.ylabel('Eye position (meters)')
plt.title('Initial Eye Position Samples')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Hand position
hand_series = nwb.processing["behavior"].data_interfaces["hand_position"]
hand_data = hand_series.data[:50000]
hand_times = hand_series.timestamps[:50000]

plt.figure(figsize=(10,5))
plt.plot(hand_times, hand_data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Initial Hand Position Samples')
plt.tight_layout()
plt.show()

# %% [markdown]
# *In the hand position plot, you see clear transitions from rest to movement indicating joystick dynamics. The eye traces above appeared contaminated with noise/artifacts during this initial segment. Nevertheless, access methods apply for further/better data selections.*
#
# **Tip:** For your own analyses, consider filtering for noise reduction or focusing on subsets corresponding to specific trial intervals found in `nwb.trials` or `nwb.processing` modules.

# %% [markdown]
# ## About the NWB file contents
#
# - **Units (spikes)**: Located under `nwb.processing["ecephys"].data_interfaces["units"]`
# - **LFP / extracellular recording**: Found via `nwb.acquisition` or processing data
# - **Trials info**: In `nwb.trials`
# - **Electrode metadata**: `nwb.electrodes` with columns such as location, group, gain etc.
#
# For more advanced analysis, consider:
#
# - Visualizing spiking data across units
# - Aligning behavior with spiking or LFP signals
# - Extracting trial-wise metrics from `nwb.trials`

# %% [markdown]
# ## Conclusion
#
# This notebook illustrated how to access Dandiset 001275, load NWB data remotely, and visualize behavioral signals. It provides a foundation for deeper analyses combining behavior, spike, and LFP data.
#
# **Remember:** This was generated automatically and might need review and refinement for more conclusive scientific insights.