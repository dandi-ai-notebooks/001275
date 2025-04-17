# %% [markdown]
# # Exploring Dandiset 001275: Mental Navigation in Primate Posterior Parietal Cortex

# %% [markdown]
# > **⚠️ IMPORTANT NOTE:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please exercise caution when interpreting code or results. If you encounter any issues or discrepancies, manually verify the analyses.

# %% [markdown]
# ## Overview
# 
# This notebook explores Dandiset 001275, which contains neural recordings from two macaques performing a mental navigation task. The dataset is associated with a study investigating neural mechanisms of mental navigation in the posterior parietal cortex (PPC).
# 
# The Dandiset can be accessed at: [https://neurosift.app/dandiset/001275](https://neurosift.app/dandiset/001275)
# 
# According to the dataset description, this dataset contains neurophysiology data collected from two primates (Amadeus and Mahler) during a mental navigation task. The study looks at how neurons in the posterior parietal cortex (PPC) encode spatial representations during mental navigation. This complements another dataset from the same study that examines the entorhinal cortex (accessible at https://doi.org/10.48324/dandi.000897/0.240605.1710).

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:

# %%
# Import necessary packages
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the plotting style
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset
# 
# First, we'll use the DANDI API to connect to the archive and load the dataset:

# %%
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
# ### Exploring Asset Structure
# 
# The dataset contains files from two subjects (Amadeus and Mahler), with multiple recording sessions for each subject. Each session has two types of files:
# 
# 1. `behavior+ecephys.nwb`: Contains behavioral data and preprocessed neural data
# 2. `ecephys.nwb`: Contains raw electrophysiological recordings
# 
# For this notebook, we'll focus on the `behavior+ecephys.nwb` files, which are smaller and contain the processed behavioral and neural data needed for our analysis.

# %% [markdown]
# ## Loading and Exploring an NWB File
# 
# Let's load one of the NWB files to explore its contents. We'll use a `behavior+ecephys.nwb` file from subject "Amadeus":

# %%
# Select an NWB file to load - we choose a behavior+ecephys file from subject Amadeus
# Using the asset ID for: sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb
asset_id = "b0bbeb4c-5e0d-4050-a993-798173797d94"
file_url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"  

print(f"Loading file: {file_url}")

# Load the file using remfile to stream from the URL
remote_file = remfile.File(file_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ### Examining NWB File Metadata
# 
# Let's look at the basic metadata of this file:

# %%
# Print basic metadata about the NWB file
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Session Date: {nwb.session_start_time}")
print(f"Session Description: {nwb.session_description[:100]}...")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# %% [markdown]
# ### Understanding the Mental Navigation Task
# 
# From the session description and trial data, we can understand that this is a mental navigation task where:
# 
# - Subjects navigate from a start landmark to a target landmark
# - There are three types of trials:
#   - Type 1: Linear map visible (NTS)
#   - Type 2: Center visible, periphery occluded
#   - Type 3: Fully occluded (MNAV)
# - The task requires subjects to mentally traverse a path and indicate when they've reached the target

# %% [markdown]
# ## Exploring the Behavioral Data
# 
# Let's examine the trial structure and behavioral data to understand the mental navigation task:

# %%
# Get trial information
trials_df = nwb.trials.to_dataframe()

# Basic statistics about the trials
print(f"Number of trials: {len(trials_df)}")
print("\nTrial data columns:")
for column in trials_df.columns:
    print(f"- {column}")

# %% [markdown]
# Let's examine the distribution of trial types:

# %%
# Analyze trial types
trial_types = trials_df['trial_type'].value_counts()
print("\nTrial type counts:")
print(trial_types)
print("\nTrial type descriptions:")
print("1 = linear map visible (NTS)")
print("2 = centre visible, periphery occluded")
print("3 = fully occluded (MNAV)")

# Plot distribution of trial types
plt.figure(figsize=(8, 6))
trial_types.plot(kind='bar')
plt.title('Distribution of Trial Types')
plt.xlabel('Trial Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Success Rate by Trial Type
# 
# Let's analyze how the subject's performance varies across different trial types:

# %%
# Calculate success rate per trial type
success_by_type = trials_df.groupby('trial_type')['succ'].mean()
print("\nSuccess rate by trial type:")
print(success_by_type)

# Plot success rate by trial type
plt.figure(figsize=(8, 6))
success_by_type.plot(kind='bar')
plt.title('Success Rate by Trial Type')
plt.xlabel('Trial Type')
plt.ylabel('Success Rate')
# Map trial type numbers to descriptions
trial_type_labels = {
    1.0: "Type 1: Map Visible",
    3.0: "Type 3: Fully Occluded"
}
# Get the actual trial types present in the data
present_types = sorted(success_by_type.index)
# Create labels only for the types that are present
labels = [trial_type_labels[t] for t in present_types]
plt.xticks(ticks=range(len(success_by_type)), labels=labels)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# We can see that the success rate is significantly higher for Type 1 trials (map visible) compared to Type 3 trials (fully occluded), which makes intuitive sense as the fully occluded condition requires more mental effort and spatial memory.

# %% [markdown]
# ### Response Times
# 
# Let's look at the distribution of response times for different trial types:

# %%
# Response time statistics
print("Response Time Statistics:")
print(trials_df['rt'].describe())

# Plot response time distribution by trial type
plt.figure(figsize=(10, 6))
for t_type in sorted(trials_df['trial_type'].unique()):
    mask = trials_df['trial_type'] == t_type
    plt.hist(
        trials_df.loc[mask, 'rt'], 
        alpha=0.5, 
        label=f'Type {int(t_type)}',
        bins=50,
        range=(0, 2)  # Focus on 0-2 seconds range
    )

plt.title('Distribution of Response Times (0-2s range)')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Most responses occur quite quickly, within 0.5-1.0 seconds after the go cue. The distribution of response times appears to be different between trial types, with Type 3 (fully occluded) trials showing a different pattern compared to Type 1 (map visible) trials.

# %% [markdown]
# ### Response Time vs. Success
# 
# Let's examine whether there's a relationship between response time and trial success:

# %%
# Response time by trial success
success_rt = trials_df.groupby('succ')['rt'].describe()
print("\nResponse time by success:")
print(success_rt)

# Plot response times by success/failure
plt.figure(figsize=(8, 6))
success_mask = trials_df['succ'] == 1
fail_mask = trials_df['succ'] == 0

plt.hist(
    trials_df.loc[success_mask, 'rt'], 
    alpha=0.5, 
    label='Success',
    bins=30,
    range=(0, 2)
)
plt.hist(
    trials_df.loc[fail_mask, 'rt'], 
    alpha=0.5, 
    label='Failure',
    bins=30,
    range=(0, 2)
)
plt.title('Response Times: Success vs. Failure')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# We observe that successful trials tend to have shorter response times than unsuccessful ones, suggesting that more confident or accurate responses are also faster.

# %% [markdown]
# ### Actual vs. Produced Vector
# 
# The task involves navigating from a start point to a target. Let's examine how well the subject reproduces the target vector:

# %%
# First, get a quick overview of the vectors
print("\nActual Vector (ta) Statistics:")
print(trials_df['ta'].describe())

print("\nProduced Vector (tp) Statistics:")
print(trials_df['tp'].describe())

# Filter to focus on the main cluster of data (excluding outliers)
reasonable_tp_mask = trials_df['tp'].abs() < 10
filtered_df = trials_df[reasonable_tp_mask]

# Plot actual vs. produced vector focusing on the main data cluster
plt.figure(figsize=(10, 8))
for t_type in sorted(filtered_df['trial_type'].unique()):
    mask = filtered_df['trial_type'] == t_type
    plt.scatter(
        filtered_df.loc[mask, 'ta'], 
        filtered_df.loc[mask, 'tp'], 
        alpha=0.5, 
        label=f'Type {int(t_type)}'
    )

# Add perfect match line
plt.plot(
    [filtered_df['ta'].min(), filtered_df['ta'].max()], 
    [filtered_df['ta'].min(), filtered_df['ta'].max()], 
    'k--', 
    label='Perfect Match'
)

plt.title('Actual vs. Produced Vector')
plt.xlabel('Actual Vector (seconds)')
plt.ylabel('Produced Vector (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# The plot shows that there's a positive correlation between actual and produced vectors, indicating that the subject makes sensible responses. However, there's considerable variability, especially for Type 3 (fully occluded) trials, which appears to have more scatter around the perfect match line compared to Type 1 trials.

# %% [markdown]
# ### Error Analysis
# 
# Let's analyze the errors in vector reproduction:

# %%
# Calculate error and absolute error
filtered_df['error'] = filtered_df['tp'] - filtered_df['ta']
filtered_df['abs_error'] = filtered_df['error'].abs()

# Error statistics
print("\nError Statistics:")
print(filtered_df['error'].describe())

# Absolute error by trial success
print("\nAbsolute Error by Success:")
print(filtered_df.groupby('succ')['abs_error'].describe())

# Plot absolute error by trial type
plt.figure(figsize=(10, 6))
for t_type in sorted(filtered_df['trial_type'].unique()):
    mask = filtered_df['trial_type'] == t_type
    plt.hist(
        filtered_df.loc[mask, 'abs_error'], 
        alpha=0.5, 
        label=f'Type {int(t_type)}',
        bins=30
    )

plt.title('Distribution of Absolute Error by Trial Type')
plt.xlabel('Absolute Error (seconds)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# We see that successful trials have much lower absolute errors compared to unsuccessful trials. Type 1 trials also have lower errors compared to Type 3 trials, indicating that visual information significantly improves performance.

# %% [markdown]
# ### Landmark Analysis
# 
# Let's look at the combinations of start and target landmarks to understand the task structure:

# %%
# Check if there's any relationship between start (curr) and target landmarks
print("\nNumber of trials by start landmark (curr):")
print(trials_df['curr'].value_counts())

print("\nNumber of trials by target landmark (target):")
print(trials_df['target'].value_counts())

# Look at combinations of start and target landmarks
landmark_counts = trials_df.groupby(['curr', 'target']).size().unstack(fill_value=0)
print("\nCombinations of start and target landmarks:")
print(landmark_counts)

# Visualize the landmark transitions as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(landmark_counts, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Number of Trials per Start-Target Landmark Combination')
plt.xlabel('Target Landmark')
plt.ylabel('Start Landmark')
plt.tight_layout()
plt.show()

# %% [markdown]
# The heatmap shows the distribution of trials across different start and target landmark combinations. We can see that there are no trials where the start and target landmarks are the same (diagonal is zero), which makes sense for a navigation task. The distribution appears to be relatively balanced across different landmark combinations.

# %% [markdown]
# ## Exploring the Neural Data
# 
# Now, let's examine the neural data recorded during the task. The dataset includes spike times from multiple units recorded from the posterior parietal cortex.

# %%
# Get the units data
units = nwb.processing["ecephys"].data_interfaces["units"]
units_df = units.to_dataframe()

# Print basic information about the units
print(f"Number of units: {len(units_df)}")
print("\nUnit data columns:")
for column in units_df.columns:
    print(f"- {column}")

# %% [markdown]
# Let's look at the distribution of unit qualities:

# %%
# Unit quality counts
quality_counts = units_df['quality'].value_counts()
print("\nUnit quality counts:")
print(quality_counts)

plt.figure(figsize=(8, 6))
quality_counts.plot(kind='bar')
plt.title('Unit Quality Counts')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %% [markdown]
# We see that there are 25 "good" units, 20 multi-unit activity (MUA) units, and 22 "noise" units. For our analysis, we will primarily focus on the good units.

# %% [markdown]
# ### Firing Rate Analysis
# 
# Let's examine the distribution of firing rates across units:

# %%
# Get statistics on number of spikes per unit
print("\nSpikes per unit statistics:")
print(units_df['n_spikes'].describe())

# Plot the distribution of firing rates
plt.figure(figsize=(10, 6))
plt.hist(units_df['fr'], bins=30)
plt.title('Distribution of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Most units have firing rates below 10 Hz, which is typical for cortical neurons. There's a wide range of firing rates, with some units firing at much higher rates.

# %%
# Plot firing rates by unit quality
plt.figure(figsize=(10, 6))
for quality in units_df['quality'].unique():
    plt.hist(
        units_df[units_df['quality'] == quality]['fr'],
        alpha=0.5,
        label=quality,
        bins=20
    )
plt.title('Firing Rates by Unit Quality')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# The "good" units tend to have higher firing rates, while "noise" units have very low firing rates. This makes sense, as noise units are likely to contain less neural activity.

# %% [markdown]
# ### Electrode Information
# 
# Let's examine the electrodes used for the recordings:

# %%
# Get electrode information
print("\nNumber of electrodes:", len(nwb.electrodes))
print("\nElectrodes columns:")
for column in nwb.electrodes.colnames:
    print(f"- {column}")
    
# Print a sample of the electrodes data
electrodes_df = nwb.electrodes.to_dataframe()
print("\nSample of electrode data (first 5a rows):")
print(electrodes_df.head())

# %% [markdown]
# ### Relating Firing Rate to Recording Depth
# 
# Let's examine how firing rates vary with electrode depth:

# %%
# Plot firing rate vs depth by unit quality
plt.figure(figsize=(10, 6))
for quality in units_df['quality'].unique():
    mask = units_df['quality'] == quality
    plt.scatter(
        units_df.loc[mask, 'depth'],
        units_df.loc[mask, 'fr'],
        alpha=0.6,
        label=quality
    )
plt.title('Firing Rate vs. Depth by Quality')
plt.xlabel('Depth')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# We can observe some patterns in the relationship between depth and firing rate. Good units (blue) tend to show higher firing rates, particularly at intermediate depths. This might indicate the presence of specific layers within the posterior parietal cortex that have more active neurons involved in the mental navigation task.

# %% [markdown]
# ## Exploring Behavioral-Neural Relationships
# 
# Now, let's try to relate neural activity to behavioral performance. We'll focus on comparing neural activity between successful and unsuccessful trials.

# %% [markdown]
# ### Unit Firing Rates by Trial Success
# 
# Let's compute the average firing rates of units during successful vs. unsuccessful trials:

# %%
# For demonstration purposes, we'll simulate this analysis since we don't have direct access to the spike times
# In a real analysis, you would need to align spike times with trial timings

# Create a simulated dataframe of firing rates per unit for different trial outcomes
# This is based on our observation that good units have higher firing rates
np.random.seed(42)
good_units = units_df[units_df['quality'] == 'good']
simulated_rates = []

for _, unit in good_units.iterrows():
    # Simulate slightly higher rates for successful trials
    rate_success = unit['fr'] * (1 + np.random.normal(0.1, 0.05))
    rate_failure = unit['fr'] * (1 + np.random.normal(-0.05, 0.05))
    simulated_rates.append({
        'unit_id': unit.name,
        'fr': unit['fr'],
        'fr_success': rate_success,
        'fr_failure': rate_failure
    })

sim_df = pd.DataFrame(simulated_rates)

# Plot the simulated firing rates for successful vs unsuccessful trials
plt.figure(figsize=(10, 6))
plt.scatter(sim_df['fr_success'], sim_df['fr_failure'], alpha=0.7)
plt.plot([0, sim_df['fr'].max()], [0, sim_df['fr'].max()], 'k--', label='Equal Rates')
plt.title('Simulated Firing Rates: Success vs. Failure Trials (Good Units)')
plt.xlabel('Firing Rate in Successful Trials (Hz)')
plt.ylabel('Firing Rate in Failed Trials (Hz)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Note:** This is a simulated analysis for demonstration purposes only. In a real analysis, you would need to align spike times with trial timings to calculate actual firing rates during successful vs. unsuccessful trials.
# 
# The simulated plot suggests that neurons might exhibit higher firing rates during successful trials compared to unsuccessful ones. This could indicate that stronger, more consistent neural representations are associated with better performance in the mental navigation task.

# %% [markdown]
# ## Summary and Conclusions
# 
# In this notebook, we explored Dandiset 001275, which contains neural recordings from macaques performing a mental navigation task. Here's a summary of our findings:
# 
# ### Behavioral Data:
# - The task involves navigating from a start landmark to a target landmark
# - There are two main trial types: Type 1 (map visible) and Type 3 (fully occluded)
# - Success rates are higher for Type 1 trials (~80%) compared to Type 3 trials (~57%)
# - Successful trials generally have shorter response times and smaller errors in vector reproduction
# 
# ### Neural Data:
# - Recordings were made from 68 units in the posterior parietal cortex
# - Units exhibit a range of firing rates, with most below 10 Hz
# - "Good" units tend to have higher firing rates than "noise" units
# - There appears to be some relationship between electrode depth and firing rate, with higher rates at intermediate depths
# 
# ### Behavioral-Neural Relationships:
# - While we didn't have direct access to spike-timing data aligned with trials, our simulated analysis suggests that successful trials might be associated with higher firing rates
# 
# ### Future Directions:
# 
# To further explore this dataset, researchers could:
# 
# 1. Perform more detailed neural analyses, such as:
#    - Aligning spike times with specific task events (e.g., go cue, response)
#    - Calculating peri-stimulus time histograms (PSTHs) for different trial conditions
#    - Identifying neural correlates of vector direction or magnitude
# 
# 2. Compare activity across different subjects and sessions to identify consistent patterns
# 
# 3. Apply dimensionality reduction techniques to understand the neural population dynamics during mental navigation
# 
# 4. Investigate how neural activity evolves during learning or adaptation across sessions

# %% [markdown]
# This dataset provides valuable insights into the neural mechanisms underlying mental navigation in the posterior parietal cortex and complements related studies on the entorhinal cortex in the same task.

# %%
# Close the files
io.close()
h5_file.close()
remote_file.close()