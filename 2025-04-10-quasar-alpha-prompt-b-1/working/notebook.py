# %% [markdown]
# # Dandiset 001275 Explorer Notebook
#
# **Important:** This notebook was automatically generated using *dandi-notebook-gen* and has not been fully reviewed or verified. Please exercise caution in interpreting code and results, and consider validating all analyses and conclusions yourself.
#
# ---
#
# # Overview
#
# This notebook provides an introduction to **DANDI:001275**  
# **Title:** Mental navigation primate PPC Neupane_Fiete_Jazayeri  
# **Contributors:** Neupane, Sujaya  
# **Institution:** MIT, Jazayeri and Fiete Labs  
#
# This dataset contains neurophysiology recordings from two macaque monkeys performing a *mental navigation* task associated with this study: [Nature (2024)](https://doi.org/10.1038/s41586-024-07557-z).
#
# It includes:
# - Extracellular electrophysiology data
# - Behavioral signals (eye, hand movement)
# - Spike sorting results
# - Metadata on sessions, electrodes, trials
#
# **Note:** Data is streamed from the DANDI archive, so be mindful when loading large files or entire datasets.

# %% [markdown]
# ## Accessing the Dandiset assets
#
# Below we demonstrate how to use the DANDI API to list all assets/files in Dandiset `001275`:

# %%
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001275", "draft")
assets = list(dandiset.get_assets())
print(f"Total assets in Dandiset 001275: {len(assets)}")
print("First 5 asset paths:")
for a in assets[:5]:
    print("-", a.path)

# %% [markdown]
# ## Selecting NWB files to explore
#
# This Dandiset contains multiple `.nwb` files, including recordings from different sessions and subjects.
# Full list of filenames was shown above.
#
# In this example, we will focus on one of the smaller files containing both behavioral and electrophysiology data:
#
# `sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb`

# %% [markdown]
# ## Loading NWB file via PyNWB + remfile
#
# The cell below shows how to stream the NWB file remotely via DANDI without downloading the entire file locally:

# %%
import pynwb
import h5py
import remfile

nwb_url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
file = remfile.File(nwb_url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwbfile = io.read()

print("Session ID:", nwbfile.session_id)
print("Subject ID:", nwbfile.subject.subject_id if nwbfile.subject else "N/A")
print("Start time:", nwbfile.session_start_time)
if hasattr(nwbfile, 'lab'):
    print("Lab:", nwbfile.lab)
if hasattr(nwbfile, 'institution'):
    print("Institution:", nwbfile.institution)

# %% [markdown]
# ## NWB file content overview
#
# Let's examine the key components of the NWB container, such as:
# - Trials
# - Units (spike times)
# - Electrodes
# - Processing modules (e.g., behavior, ecephys)

# %%
print("Available intervals (TimeIntervals):", list(nwbfile.intervals.keys()))
print("Available processing modules:", list(nwbfile.processing.keys()))

if "ecephys" in nwbfile.processing:
    ecephys_mod = nwbfile.processing["ecephys"]
    print("Ecephys module contents:", list(ecephys_mod.data_interfaces.keys()))

if "behavior" in nwbfile.processing:
    beh_mod = nwbfile.processing["behavior"]
    print("Behavior module contents:", list(beh_mod.data_interfaces.keys()))

# %% [markdown]
# ## Inspect spikes and unit information
#
# We can explore sorted units, their IDs, spike times, and (optionally) waveforms.

# %%
if "ecephys" in nwbfile.processing and "units" in nwbfile.processing["ecephys"].data_interfaces:
    units = nwbfile.processing["ecephys"].data_interfaces["units"]
    unit_ids = units.id[:]
    print(f"Total units: {len(unit_ids)}")
    print("First 10 unit IDs:", unit_ids[:10])

    # Example: spike times for first unit (if available)
    if len(unit_ids) > 0:
        spike_train = units["spike_times"][0][:]
        print(f"Spike train length of unit 0: {len(spike_train)}")
        print("First 10 spikes (s):", spike_train[:10])

# %% [markdown]
# ## Trials data
#
# Access the behavioral **trial events**:

# %%
if hasattr(nwbfile, 'trials') and nwbfile.trials is not None:
    print("Trial columns:", nwbfile.trials.colnames)
    print("Number of trials:", len(nwbfile.trials.id))
    print("First 5 trial IDs:", nwbfile.trials.id[:5])

# %% [markdown]
# ## Accessing behavioral timeseries: eye and hand position
#
# Behavior data may be found under the "behavior" processing module.

# %%
try:
    beh_mod = nwbfile.processing.get("behavior", None)
    if beh_mod:
        eye = beh_mod.data_interfaces.get("eye_position", None)
        hand = beh_mod.data_interfaces.get("hand_position", None)

        if eye:
            print("Eye position data shape:", eye.data.shape)
            print("Eye timestamps shape:", eye.timestamps.shape)
            print(f"Eye position units: {eye.unit}")

        if hand:
            print("Hand position data shape:", hand.data.shape)
            print("Hand timestamps shape:", hand.timestamps.shape)
            print(f"Hand position units: {hand.unit}")

except Exception as e:
    print(f"Error loading behavioral data: {e}")

# %% [markdown]
# ## Plotting examples (customize with your own data inspection)
#
# Below code demonstrates *how* you might visualize behavioral data.
# However, the explored sample file contained no continuous eye or hand data in the snippets.
#
# Replace the indices or conditions as appropriate to your dataset.

# %%
import matplotlib.pyplot as plt

try:
    # Example: select snippet indices with data
    # eye_times = eye.timestamps[:]
    # eye_data = eye.data[:]
    # plt.plot(eye_times, eye_data[:,0], label='Eye X')
    # plt.plot(eye_times, eye_data[:,1], label='Eye Y')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Eye position (meters)")
    # plt.legend()
    # plt.show()

    pass  # remove this and uncomment above as needed when data is available

except Exception as e:
    print(f"No behavioral snippet to plot: {e}")

# %% [markdown]
# ## Summary and next steps
#
# This notebook demonstrated how to:
# - Connect to DANDI and list assets
# - Stream NWB files remotely
# - Access basic metadata: trials, units, electrodes
# - Explore the data hierarchy interactively
#
# **Next steps for researchers:**
# - Select sessions/files of interest
# - Explore spike data, LFP, behavioral traces in more depth
# - Perform unit quality control, waveform analysis
# - Cross-reference behavioral events and neural activity
#
# **Reminder:** This notebook is AI-generated and intended as a scaffold. Please customize, validate, and extend as needed for your scientific analyses.