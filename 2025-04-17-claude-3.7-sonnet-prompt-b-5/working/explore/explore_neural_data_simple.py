# This script explores the neural metadata from the NWB file
# We'll focus on the unit properties without attempting to access raw spike times

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

# Get statistics on number of spikes per unit
print("\nSpikes per unit statistics:")
print(units_df['n_spikes'].describe())

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
plt.savefig('explore/firing_rates_by_quality.png')

# Look at the relationship between firing rate and contamination percentage
plt.figure(figsize=(10, 6))
plt.scatter(units_df['ContamPct'], units_df['fr'], alpha=0.6)
plt.title('Firing Rate vs. Contamination Percentage')
plt.xlabel('Contamination Percentage')
plt.ylabel('Firing Rate (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rate_vs_contamination.png')

# Look at the relationship between amplitude and firing rate
plt.figure(figsize=(10, 6))
plt.scatter(units_df['Amplitude'], units_df['fr'], alpha=0.6)
plt.title('Firing Rate vs. Amplitude')
plt.xlabel('Amplitude')
plt.ylabel('Firing Rate (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rate_vs_amplitude.png')

# Look at the relationship between depth and firing rate
plt.figure(figsize=(10, 6))
plt.scatter(units_df['depth'], units_df['fr'], alpha=0.6)
plt.title('Firing Rate vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Firing Rate (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rate_vs_depth.png')

# Look at the relationship between depth and unit quality
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
plt.savefig('explore/firing_rate_vs_depth_by_quality.png')

# Get electrode information
print("\nNumber of electrodes:", len(nwb.electrodes))
print("\nElectrodes columns:")
for column in nwb.electrodes.colnames:
    print(f"- {column}")
    
# Print a sample of the electrodes data
electrodes_df = nwb.electrodes.to_dataframe()
print("\nSample of electrode data (first 5 rows):")
print(electrodes_df.head())

# Clean up
io.close()
h5_file.close()
remote_file.close()

print("\nNeural data exploration completed and figures saved!")