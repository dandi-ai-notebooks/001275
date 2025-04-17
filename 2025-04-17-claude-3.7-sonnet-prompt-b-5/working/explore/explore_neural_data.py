# This script explores the neural data from the NWB file
# We'll focus on the units (neurons) recorded during the mental navigation task

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the units data
units = nwb.processing["ecephys"].data_interfaces["units"]
units_df = units.to_dataframe()

# Print basic information about the units
print(f"Number of units: {len(units_df)}")
print("\nUnit data columns:")
for column in units_df.columns:
    print(f"- {column}")

# Print a sample of the units data
print("\nSample of unit data (first 5 rows):")
print(units_df.head())

# Look at the distribution of firing rates
plt.figure(figsize=(10, 6))
plt.hist(units_df['fr'], bins=30)
plt.title('Distribution of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rate_distribution.png')

# Look at the quality of units
quality_counts = units_df['quality'].value_counts()
print("\nUnit quality counts:")
print(quality_counts)

plt.figure(figsize=(8, 6))
quality_counts.plot(kind='bar')
plt.title('Unit Quality Counts')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('explore/unit_quality_counts.png')

# Get trial data for alignment
trials_df = nwb.trials.to_dataframe()

# Pick a random unit with a reasonable number of spikes for visualization
# Sort by number of spikes and pick one with a good amount
sorted_units = units_df.sort_values('n_spikes', ascending=False)

# Select one of the top units to visualize
target_unit_id = sorted_units.iloc[0].name
print(f"\nSelected unit {target_unit_id} with {sorted_units.iloc[0]['n_spikes']} spikes for visualization")

# Get spike times for the selected unit
# For NWB format, we need to access the spike times using the spike_times_index
unit_spike_times_idx = units.spike_times_index[target_unit_id]
# Get the actual spike times from the VectorData
all_spike_times = units.spike_times.data[:]
# Extract spike times for this specific unit
start_idx = 0 if target_unit_id == 0 else units.spike_times_index[target_unit_id - 1]
end_idx = units.spike_times_index[target_unit_id]
spike_times = all_spike_times[start_idx:end_idx]

# Print some statistics about the spike times
print(f"Number of spikes: {len(spike_times)}")
print(f"Spike time range: {spike_times.min() if len(spike_times) > 0 else 'N/A'} to {spike_times.max() if len(spike_times) > 0 else 'N/A'} seconds")

# Plot raster plot for a subset of spike times (first 60 seconds)
plt.figure(figsize=(12, 4))
max_time = 60  # First minute
if len(spike_times) > 0:
    mask = spike_times < max_time
    plt.vlines(spike_times[mask], 0, 1, color='black')
plt.title(f'Raster Plot for Unit {target_unit_id} (First {max_time} seconds)')
plt.xlabel('Time (s)')
plt.yticks([])
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('explore/unit_raster_plot.png')

# Calculate and plot interspike interval (ISI) histogram
if len(spike_times) > 1:
    isis = np.diff(spike_times)
    plt.figure(figsize=(10, 6))
    plt.hist(isis, bins=100, range=(0, 0.5))
    plt.title(f'Interspike Interval Distribution for Unit {target_unit_id}')
    plt.xlabel('Interspike Interval (s)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('explore/unit_isi_histogram.png')
else:
    print(f"Insufficient spikes for Unit {target_unit_id} to calculate ISI histogram")

# Skip trial-aligned activity if no spikes
if len(spike_times) == 0:
    print("Skipping trial-aligned activity due to insufficient spike data")
else:
    # Look at trial-aligned activity for a specific subset of trials
    # Filter for successful trials of type 3 (fully occluded)
    successful_type3_trials = trials_df[(trials_df['trial_type'] == 3.0) & (trials_df['succ'] == 1)]

    # Select a small subset of trials for visualization
    trial_subset = successful_type3_trials.head(20)

    # Initialize figure for trial-aligned activity
    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Create spike raster plot for each trial
    ax1 = plt.subplot(gs[0])
    for i, (_, trial) in enumerate(trial_subset.iterrows()):
        # Extract trial timing information
        go_cue_time = trial['gocuettl']
        response_time = trial['joy1onttl']
        
        # Find spikes within a window around the go cue (-0.5s to +2s)
        window_start = go_cue_time - 0.5
        window_end = go_cue_time + 2
        trial_spikes = spike_times[(spike_times >= window_start) & (spike_times <= window_end)]
        
        # Align spike times to go cue (time 0)
        aligned_spikes = trial_spikes - go_cue_time
        
        # Plot spikes
        ax1.vlines(aligned_spikes, i, i+0.9, color='black', lw=0.5)
        
        # Mark response time
        response_aligned = response_time - go_cue_time
        if response_aligned <= 2:  # Only mark if within the window
            ax1.plot(response_aligned, i+0.5, 'ro', markersize=4)

    # Add lines marking events
    ax1.axvline(0, color='blue', linestyle='--', label='Go Cue')
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylabel('Trial #')
    ax1.set_title(f'Unit {target_unit_id} Activity Aligned to Go Cue')
    ax1.legend()

    # Create PSTH in bottom panel
    ax2 = plt.subplot(gs[1], sharex=ax1)
    all_spikes = []
    for _, trial in trial_subset.iterrows():
        go_cue_time = trial['gocuettl']
        window_start = go_cue_time - 0.5
        window_end = go_cue_time + 2
        trial_spikes = spike_times[(spike_times >= window_start) & (spike_times <= window_end)]
        aligned_spikes = trial_spikes - go_cue_time
        all_spikes.extend(aligned_spikes)

    # Create PSTH with 50ms bins
    bins = np.arange(-0.5, 2.01, 0.05)
    counts, edges = np.histogram(all_spikes, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    rate = counts / len(trial_subset) / 0.05  # Convert to spikes/s

    ax2.bar(centers, rate, width=0.05, color='black', alpha=0.7)
    ax2.axvline(0, color='blue', linestyle='--')
    ax2.set_xlabel('Time from Go Cue (s)')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_xlim(-0.5, 2)
    ax2.grid(True)

plt.tight_layout()
plt.savefig('explore/unit_trial_aligned_activity.png')

# Clean up
io.close()
h5_file.close()
remote_file.close()

print("\nNeural data exploration completed and figures saved!")