# This script explores the behavioral data from the NWB file in more detail
# We'll focus on understanding the response times and trial structure more thoroughly

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial information
trials_df = nwb.trials.to_dataframe()

# First, let's examine the response time data more closely
print("Response Time Statistics:")
print(trials_df['rt'].describe())

# Plot a more detailed histogram of response times with appropriate binning
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
plt.savefig('explore/response_time_distribution_detailed.png')

# Let's also check if there's any correlation between response time and success
success_rt = trials_df.groupby('succ')['rt'].describe()
print("\nResponse time by success:")
print(success_rt)

# Plot success vs. failure response times
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
plt.savefig('explore/rt_success_vs_failure.png')

# Now, let's take a closer look at the actual vs. produced vector data
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

plt.title('Actual vs. Produced Vector (Filtered Data)')
plt.xlabel('Actual Vector (seconds)')
plt.ylabel('Produced Vector (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/actual_vs_produced_vector_filtered.png')

# Let's also look at the correlation between error and success
filtered_df['error'] = filtered_df['tp'] - filtered_df['ta']
filtered_df['abs_error'] = filtered_df['error'].abs()

print("\nError Statistics:")
print(filtered_df['error'].describe())

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
plt.savefig('explore/absolute_error_by_type.png')

# Check if there's any relationship between start (curr) and target landmarks
print("\nNumber of trials by start landmark (curr):")
print(trials_df['curr'].value_counts())

print("\nNumber of trials by target landmark (target):")
print(trials_df['target'].value_counts())

# Look at combinations of start and target landmarks
landmark_counts = trials_df.groupby(['curr', 'target']).size().unstack(fill_value=0)
print("\nCombinations of start and target landmarks:")
print(landmark_counts)

# Clean up
io.close()
h5_file.close()
remote_file.close()

print("\nDetailed behavioral exploration completed and figures saved!")