# %% [markdown]
# # Exploring Dandiset 001275: Mental Navigation in Primate PPC

# %% [markdown]
# **⚠️ CAUTION: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results. ⚠️**

# %% [markdown]
# ## Overview
# 
# This notebook explores Dandiset 001275, which contains neurophysiology data collected from two primates (Amadeus and Mahler) during a mental navigation task. The data is associated with the study published at https://doi.org/10.1038/s41586-024-07557-z.
# 
# You can view this dataset on Neurosift: https://neurosift.app/dandiset/001275
# 
# In this notebook, we will:
# 1. Load the Dandiset and examine its structure
# 2. Explore metadata from a specific NWB file
# 3. Visualize behavioral data (eye position and hand position)
# 4. Examine neural activity and its relationship to behavior
# 5. Analyze trial information from the mental navigation task

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import remfile
import pynwb
from dandi.dandiapi import DandiAPIClient

import seaborn as sns
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset
# 
# We'll start by connecting to the DANDI archive and retrieving information about the dataset.

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Exploring a Specific NWB File
# 
# For this analysis, we'll focus on a behavior+ecephys NWB file from subject Amadeus. This file contains both behavioral data and electrophysiology recordings from a single session.

# %%
# URL for the NWB file we'll be analyzing
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %%
# Display basic metadata about the file
print(f"NWB File: {nwb.identifier}")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Session ID: {nwb.session_id}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print(f"Session Description: {nwb.session_description}")

# %% [markdown]
# ## Understanding the Experimental Design
# 
# This experiment involves a mental navigation task where the subject (a macaque) is instructed to navigate between landmarks. Let's examine the trial structure to better understand the experimental design.

# %%
# Get trial information
trials_df = nwb.trials.to_dataframe()

# Display the first few trials
print(f"Total number of trials: {len(trials_df)}")
trials_df.head()

# %%
# Explanation of trial types
trial_types = {
    1: "Linear map visible (NTS)",
    2: "Centre visible, periphery occluded",
    3: "Fully occluded (MNAV)"
}

for type_id, description in trial_types.items():
    count = (trials_df['trial_type'] == type_id).sum()
    percent = (count / len(trials_df)) * 100
    print(f"Trial Type {type_id} ({description}): {count} trials ({percent:.1f}%)")

# %%
# Plot distribution of trial outcomes
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
success_counts = trials_df['succ'].value_counts()
plt.pie(success_counts, labels=['Failure', 'Success'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Trial Success Rate')

plt.subplot(1, 2, 2)
valid_counts = trials_df['validtrials_mm'].value_counts()
plt.pie(valid_counts, labels=['Invalid', 'Valid'], autopct='%1.1f%%', colors=['#ffcc99','#99ff99'])
plt.title('Valid Trials (GMM classification)')

plt.tight_layout()

# %% [markdown]
# ## Behavioral Data: Eye and Hand Position
# 
# Let's visualize the eye and hand position data to understand the behavioral aspect of the experiment.

# %%
# Extract eye position data (sampling a subset to manage memory)
eye_data = nwb.processing["behavior"].data_interfaces["eye_position"]
sample_size = 100000  # Sample size to avoid loading too much data
step = len(eye_data.timestamps) // sample_size

# Get the sampled data
eye_timestamps = eye_data.timestamps[::step]
eye_positions = eye_data.data[::step, :]

print(f"Eye position data: {eye_data.data.shape} total samples")
print(f"Sampled {len(eye_timestamps)} points for visualization")
print(f"Conversion factor: {eye_data.conversion} {eye_data.unit}")
print(f"Reference frame: {eye_data.reference_frame}")

# %%
# Plot eye position over time
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(eye_timestamps, eye_positions[:, 0], 'b-', alpha=0.5, label='X position')
plt.xlabel('Time (s)')
plt.ylabel(f'X Position ({eye_data.unit})')
plt.title('Eye X Position Over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(eye_timestamps, eye_positions[:, 1], 'r-', alpha=0.5, label='Y position')
plt.xlabel('Time (s)')
plt.ylabel(f'Y Position ({eye_data.unit})')
plt.title('Eye Y Position Over Time')
plt.grid(True)

plt.tight_layout()

# %%
# 2D plot of eye positions
plt.figure(figsize=(10, 10))
plt.scatter(eye_positions[:, 0], eye_positions[:, 1], alpha=0.01, s=1)
plt.xlabel(f'X Position ({eye_data.unit})')
plt.ylabel(f'Y Position ({eye_data.unit})')
plt.title('Eye Position Heatmap')
plt.axis('equal')
plt.grid(True)

# Create a hexbin plot to better visualize density
plt.figure(figsize=(10, 10))
h = plt.hexbin(eye_positions[:, 0], eye_positions[:, 1], gridsize=50, cmap='viridis')
plt.xlabel(f'X Position ({eye_data.unit})')
plt.ylabel(f'Y Position ({eye_data.unit})')
plt.title('Eye Position Density')
plt.colorbar(h, label='Count')
plt.axis('equal')

# %%
# Extract hand position data (sampling a subset to manage memory)
hand_data = nwb.processing["behavior"].data_interfaces["hand_position"]
sample_size = 100000  # Sample size to avoid loading too much data
step = len(hand_data.timestamps) // sample_size

# Get the sampled data
hand_timestamps = hand_data.timestamps[::step]
hand_positions = hand_data.data[::step]

print(f"Hand position data: {hand_data.data.shape} total samples")
print(f"Sampled {len(hand_timestamps)} points for visualization")
print(f"Conversion factor: {hand_data.conversion} {hand_data.unit}")
print(f"Reference frame: {hand_data.reference_frame}")

# %%
# Plot hand position over time
plt.figure(figsize=(12, 6))
plt.plot(hand_timestamps, hand_positions, 'g-', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel(f'Hand Position ({hand_data.unit})')
plt.title('Hand Position Over Time')
plt.grid(True)

# Create a histogram of hand positions
plt.figure(figsize=(10, 6))
plt.hist(hand_positions, bins=50, alpha=0.7)
plt.xlabel(f'Hand Position ({hand_data.unit})')
plt.ylabel('Frequency')
plt.title('Distribution of Hand Positions')
plt.grid(True)

# %% [markdown]
# ## Neural Data: Exploring Units
# 
# Now let's examine the neural data recorded from the posterior parietal cortex (PPC) during this task.

# %%
# Get units information
units = nwb.processing["ecephys"].data_interfaces["units"]
units_df = units.to_dataframe()

print(f"Total number of units: {len(units_df)}")
print(f"Columns in units dataframe: {units_df.columns.tolist()}")

# Display a summary of the units
units_df[['unit_name', 'quality', 'n_spikes', 'fr']].head(10)

# %%
# Analyze unit quality
quality_counts = units_df['quality'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts.values)
plt.xlabel('Unit Quality')
plt.ylabel('Count')
plt.title('Distribution of Unit Quality')
plt.grid(axis='y')

# %%
# Analyze firing rates
plt.figure(figsize=(10, 6))
sns.histplot(units_df['fr'], bins=20, kde=True)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Firing Rates Across Units')
plt.grid(True)

# %%
# Analyze spike counts
plt.figure(figsize=(10, 6))
sns.histplot(units_df['n_spikes'], bins=20, kde=True)
plt.xlabel('Number of Spikes')
plt.ylabel('Count')
plt.title('Distribution of Spike Counts Across Units')
plt.grid(True)

# %% [markdown]
# ## Examining Spike Timing for a Single Unit
# 
# Let's look at the spike times for one of the units and examine its activity pattern.

# %%
# Choose a unit with good quality and a reasonable number of spikes
good_units = units_df[units_df['quality'] == 'good'].sort_values(by='n_spikes', ascending=False)
if len(good_units) > 0:
    example_unit_id = good_units.index[0]
else:
    # If no good units, pick one with the most spikes
    example_unit_id = units_df.sort_values(by='n_spikes', ascending=False).index[0]

# Get information about this unit
example_unit = units_df.loc[example_unit_id]
print(f"Selected Unit ID: {example_unit_id}")
print(f"Unit Name: {example_unit['unit_name']}")
print(f"Quality: {example_unit['quality']}")
print(f"Firing Rate: {example_unit['fr']} Hz")
print(f"Number of Spikes: {example_unit['n_spikes']}")
print(f"Depth: {example_unit['depth']}")
print(f"Channel: {example_unit['ch']}")

# %%
# Get spike times for this unit
spike_times = units.spike_times_index[example_unit_id]
print(f"Number of spikes: {len(spike_times)}")

# Plot spike times
plt.figure(figsize=(14, 6))
plt.plot(spike_times, np.ones_like(spike_times), '|', markersize=10, color='black')
plt.xlabel('Time (s)')
plt.ylabel('Spikes')
plt.title(f'Spike Times for Unit {example_unit_id}')
plt.xlim(0, 60)  # Look at first 60 seconds
plt.grid(True)

# %%
# Create a PSTH (Peri-Stimulus Time Histogram) around go cue
def create_psth(spike_times, event_times, window=(-1, 2), bin_size=0.05):
    """Create a PSTH around specified event times"""
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    psth = np.zeros(len(bins) - 1)
    count = 0
    
    for event_time in event_times:
        # Find spikes within the window of this event
        mask = ((spike_times >= event_time + window[0]) & 
                (spike_times <= event_time + window[1]))
        if not any(mask):
            continue
        
        relative_times = spike_times[mask] - event_time
        hist, _ = np.histogram(relative_times, bins=bins)
        psth += hist
        count += 1
    
    if count > 0:
        psth = psth / (count * bin_size)  # Convert to firing rate in Hz
    
    return psth, bins[:-1], count

# Use go cue times from valid trials
valid_trials = trials_df[trials_df['validtrials_mm'] == 1]
go_cue_times = valid_trials['gocuettl'].dropna().values

psth, time_bins, trial_count = create_psth(spike_times, go_cue_times)

plt.figure(figsize=(12, 6))
plt.bar(time_bins, psth, width=0.05, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='Go Cue')
plt.xlabel('Time Relative to Go Cue (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title(f'PSTH for Unit {example_unit_id} around Go Cue (n={trial_count} trials)')
plt.legend()
plt.grid(True)

# %% [markdown]
# ## Relating Neural Activity to Behavior
# 
# Let's examine how neural activity relates to the behavioral aspects of the task, specifically the trial type and success.

# %%
# Explore activity patterns for different trial types
trial_type_labels = [1, 2, 3]  # Different trial types
colors = ['blue', 'green', 'red']

plt.figure(figsize=(14, 8))

for i, trial_type in enumerate(trial_type_labels):
    # Get go cue times for this trial type
    type_trials = trials_df[(trials_df['trial_type'] == trial_type) & 
                            (trials_df['validtrials_mm'] == 1)]
    type_go_cues = type_trials['gocuettl'].dropna().values
    
    # Create PSTH
    psth, time_bins, trial_count = create_psth(spike_times, type_go_cues)
    
    # Plot
    plt.subplot(3, 1, i+1)
    plt.bar(time_bins, psth, width=0.05, alpha=0.7, color=colors[i])
    plt.axvline(x=0, color='black', linestyle='--', label='Go Cue')
    plt.title(f'Trial Type {trial_type}: {trial_types[trial_type]} (n={trial_count} trials)')
    plt.ylabel('Firing Rate (Hz)')
    plt.grid(True)
    
    if i == 2:  # Only add x-label to the bottom subplot
        plt.xlabel('Time Relative to Go Cue (s)')

plt.tight_layout()

# %%
# Compare neural activity between successful and failed trials
success_labels = [0, 1]  # Failed and successful trials
success_colors = ['darkred', 'darkgreen']
success_names = ['Failed', 'Successful']

plt.figure(figsize=(12, 8))

for i, succ in enumerate(success_labels):
    # Get go cue times for this success status
    succ_trials = trials_df[(trials_df['succ'] == succ) & 
                           (trials_df['validtrials_mm'] == 1)]
    succ_go_cues = succ_trials['gocuettl'].dropna().values
    
    # Create PSTH
    psth, time_bins, trial_count = create_psth(spike_times, succ_go_cues)
    
    # Plot
    plt.subplot(2, 1, i+1)
    plt.bar(time_bins, psth, width=0.05, alpha=0.7, color=success_colors[i])
    plt.axvline(x=0, color='black', linestyle='--', label='Go Cue')
    plt.title(f'{success_names[i]} Trials (n={trial_count} trials)')
    plt.ylabel('Firing Rate (Hz)')
    plt.grid(True)
    
    if i == 1:  # Only add x-label to the bottom subplot
        plt.xlabel('Time Relative to Go Cue (s)')

plt.tight_layout()

# %% [markdown]
# ## Correlating Neural Activity with Task Parameters
# 
# Let's look at how neural activity might correlate with specific task parameters like response time.

# %%
# Create a scatter plot of response time vs. firing rate in a time window after go cue
def calculate_window_fr(spike_times, event_time, window=(0, 0.5)):
    """Calculate firing rate in a specific time window around an event"""
    mask = ((spike_times >= event_time + window[0]) & 
            (spike_times <= event_time + window[1]))
    spike_count = np.sum(mask)
    window_duration = window[1] - window[0]
    return spike_count / window_duration  # Hz

# Calculate firing rates for each trial
valid_trials = trials_df[trials_df['validtrials_mm'] == 1].copy()
response_window = (0, 0.5)  # Look at 0-500ms after go cue

# This can be computationally intensive, so limit to a reasonable number of trials
max_trials = 200
if len(valid_trials) > max_trials:
    valid_trials = valid_trials.sample(max_trials, random_state=42)

firing_rates = []
for idx, trial in valid_trials.iterrows():
    if pd.isna(trial['gocuettl']) or pd.isna(trial['rt']):
        firing_rates.append(np.nan)
        continue
    fr = calculate_window_fr(spike_times, trial['gocuettl'], response_window)
    firing_rates.append(fr)

valid_trials['post_cue_fr'] = firing_rates
valid_trials = valid_trials.dropna(subset=['post_cue_fr', 'rt'])

# %%
# Plot the relationship between firing rate and response time
plt.figure(figsize=(10, 8))

for i, (trial_type, df_group) in enumerate(valid_trials.groupby('trial_type')):
    plt.scatter(df_group['post_cue_fr'], df_group['rt'], 
                label=f'Type {trial_type}: {trial_types[trial_type]}',
                alpha=0.7, s=50)

plt.xlabel(f'Firing Rate {response_window[0]}-{response_window[1]}s after Go Cue (Hz)')
plt.ylabel('Response Time (s)')
plt.title(f'Relationship between Unit {example_unit_id} Activity and Response Time')
plt.legend()
plt.grid(True)

# Calculate correlation
correlation = valid_trials['post_cue_fr'].corr(valid_trials['rt'])
plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), 
             xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# %% [markdown]
# ## Examining Electrodes and Recording Locations
# 
# Let's look at the electrode information to understand the recording setup.

# %%
# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")
electrodes_df.head()

# %%
# Visualize electrode relative positions
plt.figure(figsize=(10, 8))
plt.scatter(electrodes_df['rel_x'], electrodes_df['rel_y'], s=50, c=range(len(electrodes_df)), cmap='viridis')
for i, row in electrodes_df.iterrows():
    plt.text(row['rel_x'], row['rel_y'], str(i), fontsize=8)
plt.xlabel('Relative X Position')
plt.ylabel('Relative Y Position')
plt.title('Electrode Positions')
plt.colorbar(label='Electrode Index')
plt.grid(True)
plt.axis('equal')

# %% [markdown]
# ## Trial Parameter Analysis
# 
# Let's analyze the behavioral variables in the trial data to better understand the experimental task.

# %%
# Analyze the relationship between actual and produced vectors (ta and tp)
plt.figure(figsize=(10, 8))

for trial_type, df_group in trials_df.groupby('trial_type'):
    plt.scatter(df_group['ta'], df_group['tp'], alpha=0.3, 
                label=f'Type {trial_type}: {trial_types[trial_type]}')

plt.plot([0, trials_df['ta'].max()], [0, trials_df['ta'].max()], 'k--', label='Perfect Match')
plt.xlabel('Actual Vector (seconds)')
plt.ylabel('Produced Vector (seconds)')
plt.title('Actual vs. Produced Time Vectors')
plt.legend()
plt.grid(True)
plt.axis('equal')

# %%
# Analyze response time (rt) distribution across trial types using violin plot instead of boxplot
plt.figure(figsize=(12, 6))

# Use violinplot which is more reliable across seaborn versions
sns.violinplot(x='trial_type', y='rt', data=trials_df, 
              order=[1, 2, 3], palette='Set3', inner='box')
# Add individual points
sns.stripplot(x='trial_type', y='rt', data=trials_df.sample(min(500, len(trials_df))), 
             order=[1, 2, 3], color='black', alpha=0.3, jitter=True)

plt.xlabel('Trial Type')
plt.ylabel('Response Time (s)')
plt.title('Response Time by Trial Type')
plt.xticks([0, 1, 2], [f"{i}: {trial_types[i]}" for i in [1, 2, 3]])
plt.grid(axis='y')

# %%
# Examine the relationship between delay and response time
plt.figure(figsize=(10, 8))
sns.scatterplot(x='delay', y='rt', hue='trial_type', data=trials_df, 
                palette=['blue', 'green', 'red'], alpha=0.5)
plt.xlabel('Delay (s)')
plt.ylabel('Response Time (s)')
plt.title('Relationship Between Delay and Response Time')
plt.legend(title='Trial Type', labels=[trial_types[i] for i in [1, 2, 3]])
plt.grid(True)

# %% [markdown]
# ## Summary and Conclusions
# 
# In this notebook, we've explored the Dandiset 001275, which contains neurophysiology data from macaques performing a mental navigation task. We've:
# 
# 1. **Loaded and examined the dataset structure** using the DANDI API
# 2. **Explored behavioral data** including eye and hand positions, revealing patterns of visual attention and motor responses
# 3. **Analyzed neural activity** from the posterior parietal cortex (PPC), including:
#    - Distributions of unit quality, firing rates, and spike counts
#    - Detailed analysis of a single unit's activity patterns
#    - Neural responses to experimental events (go cues)
#    - Correlations between neural activity and behavioral measures
# 4. **Examined trial information** to understand the experimental design and task parameters
# 
# The data shows interesting patterns of neural activity related to mental navigation. PPC neurons appear to respond differently based on trial type (whether visual guidance was available) and trial outcome (success vs. failure). These findings align with the role of PPC in spatial navigation and movement planning.
# 
# ### Future Directions
# 
# Several potential analyses could extend this work:
# 
# 1. **Population-level analysis**: Examine patterns across multiple neurons to identify population coding of navigation parameters
# 2. **Temporal dynamics**: Analyze how neural representations evolve throughout the trial
# 3. **Decoding analysis**: Attempt to decode behavioral variables from neural activity
# 4. **Comparison across sessions**: Compare neural responses across different recording sessions and subjects
# 5. **Integration with EC data**: Correlate these PPC findings with the entorhinal cortex data mentioned in the dataset description
# 
# These neural recordings from non-human primates during mental navigation provide valuable insights into the neural basis of cognitive mapping and spatial navigation.