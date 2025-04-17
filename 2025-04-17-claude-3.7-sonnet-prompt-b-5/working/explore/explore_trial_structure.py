# This script explores the trial structure and behavioral data from a selected NWB file
# We'll look at the trial information to understand the mental navigation task

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# First, get a basic overview of the file
print(f"Session ID: {nwb.session_id}")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Description: {nwb.session_description[:100]}...")

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"\nNumber of trials: {len(trials_df)}")
print("\nTrial data columns:")
for column in trials_df.columns:
    print(f"- {column}")

# Print a sample of the trial data
print("\nSample of trial data (first 5 rows):")
print(trials_df.head())

# Analyze trial types
trial_types = trials_df['trial_type'].value_counts()
print("\nTrial type counts:")
print(trial_types)
print("\nTrial type descriptions:")
print("1 = linear map visible (NTS)")
print("2 = centre visible, periphery occluded")
print("3 = fully occluded (MNAV)")

# Examine success rate per trial type
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
plt.tight_layout()
plt.savefig('explore/success_rate_by_trial_type.png')

# Look at the relationship between actual vector (ta) and produced vector (tp)
plt.figure(figsize=(10, 8))
for t_type in sorted(trials_df['trial_type'].unique()):
    mask = trials_df['trial_type'] == t_type
    plt.scatter(
        trials_df.loc[mask, 'ta'], 
        trials_df.loc[mask, 'tp'], 
        alpha=0.5, 
        label=f'Type {int(t_type)}'
    )

plt.title('Actual vs. Produced Vector')
plt.xlabel('Actual Vector (seconds)')
plt.ylabel('Produced Vector (seconds)')
plt.plot([0, trials_df['ta'].max()], [0, trials_df['ta'].max()], 'k--', label='Perfect Match')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/actual_vs_produced_vector.png')

# Look at the distribution of response times
plt.figure(figsize=(10, 6))
type_colors = {1: 'blue', 2: 'green', 3: 'red'}
for t_type in sorted(trials_df['trial_type'].unique()):
    mask = trials_df['trial_type'] == t_type
    plt.hist(
        trials_df.loc[mask, 'rt'], 
        alpha=0.5, 
        label=f'Type {int(t_type)}',
        bins=30,
        color=type_colors[t_type]
    )

plt.title('Distribution of Response Times')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/response_time_distribution.png')

# Clean up
io.close()
h5_file.close()
remote_file.close()

print("\nExploration completed and figures saved!")