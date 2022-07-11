"""

This script takes the output from the TC_refinement_app.py
(final_ccf_coordinates.csv) and adds the area/coordinate information to the
probe_info.json and (units) metrics.csv files. It also extracts location info
for the stimulating electrode and saves it as stim_elec_location.json inside the
experiment's recording1 folder.
* This will overwrite any pre-existing files. *

"""
import os
import json
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.ndimage.filters import gaussian_filter1d

################################################################################
                      ### Set locations of files ###
################################################################################

### Should update these for each subject ###

# Set the location of the subject's histology folder #
histology_loc = r'B:\mouse612221\histology'

# Set the data location for each experiment day #
# Choose the folder that contains the experiment1 folder #
# Can say: None if the dataset doesn't include a 2nd recording #
data_location = {
    1: r'B:\mouse612221\test1_2022-04-05_13-10-34',
    2: None # None
}

# Enter the probe letter location for the stim electrode #
# This should match the letter used in the annotation_app #
# For example: 'A' is the stim electrode was in the probeA holder #
stim_elec_location = None # 'E'

################################################################################
################################################################################

### Template file locations and global variables ###
### Shouldn't need to change these each time ###

midline_ind = 228 ## midline of CCF, used to find ipsi/contra chs ##

structure_tree = pd.read_csv(
    '//allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/TC_CCF_template_files/structure_tree_safe_2017.csv'
)

################################################################################
################################################################################

### Update probe_info.json and metrics.csv files for each exp/probe ###

print('\nUpdating probe location files...')
for expi, dataloci in data_location.items():
    if dataloci:
        ## Find probes in experiment folder ##
        all_probe_folders = glob(dataloci + '/**/*_sorted', recursive=True)
        probe_list = [x[x.find('_sorted')-6:x.find('_sorted')] for x in all_probe_folders]
        finaldf = pd.read_csv(os.path.join(histology_loc, 'final_ccf_coordinates_exp{:d}.csv'.format(expi)))

        for prind, probei in enumerate(probe_list):
            probedf = finaldf[finaldf.probe == probei].reset_index(drop=True)
            assigned_regions = probedf['structure_id'].values
            borders = np.where(np.diff(assigned_regions) != 0)[0]
            ## Get the 3d direction of the probe ##
            probe_axis = np.vstack((probedf['A/P'].values, probedf['D/V'].values, probedf['M/L'].values)).T
            axmean = probe_axis.mean(axis=0)
            uu,dd,vv = np.linalg.svd(probe_axis - axmean)

            allch_coords = np.zeros((384,3), dtype=int)
            all_structids = np.zeros((384,), dtype=int)
            allch_areas = []
            cha = 384
            for bordi in borders:
                d_ind = probedf.index[probedf['channels'] == cha].tolist()[0]
                chb = probedf['channels'].loc[bordi]
                if chb < 0:
                    chb = 0
                    bordi = probedf.index[probedf['channels'] == chb].tolist()[0]
                find_chs = np.arange(chb, cha)
                ab_dist = np.linalg.norm(probe_axis[d_ind,:] - probe_axis[bordi,:])
                chdists = np.flip(np.linspace(0, ab_dist, len(find_chs)))
                allch_coords[find_chs,:] = np.round(vv[0] * chdists[:,np.newaxis] + probe_axis[d_ind,:]).astype(int)
                region_id = probedf['structure_id'].loc[bordi]
                all_structids[find_chs] = region_id
                if region_id == 0:
                    region_name = "null"
                else:
                    region_name = structure_tree[structure_tree.id == region_id]['acronym'].iloc[0]
                allch_areas.extend([region_name] * len(find_chs))
                if chb == 0:
                    break
                cha = find_chs[0]

            is_ipsi = allch_coords[:,2] < midline_ind
            allch_areas = list(np.flip(allch_areas)) # flip order because list is backwards

            ## Update probe_info.json file with ch locations ##
            probe_info_location = os.path.join(all_probe_folders[prind], 'probe_info.json')
            with open(probe_info_location) as read_data_file:
                probe_data = json.load(read_data_file)

            probe_data["area_ch"] = allch_areas
            probe_data["ccf_coord_ch"] = allch_coords.tolist()
            probe_data["is_ipsi_ch"] = is_ipsi.tolist()
            probe_data["ccf_resolution"] = 25

            with open(probe_info_location, 'w') as outfile:
                json.dump(probe_data, outfile, indent = 4, separators = (',', ': '))
            print(' Saved {}.'.format(probe_info_location))

            ## Update metrics.csv file with unit locations ##
            unit_info_location = os.path.join(all_probe_folders[prind], r'continuous\Neuropix-PXI-100.0\metrics.csv')
            unit_metrics = pd.read_csv(unit_info_location)
            if 'Unnamed: 0' in unit_metrics.columns:
                unit_metrics.drop('Unnamed: 0', axis=1, inplace=True)

            unit_area = []
            unit_coords = []
            unit_ipsi = []
            for ind, row in unit_metrics.iterrows():
                unit_area.append(allch_areas[row['peak_channel']])
                unit_coords.append(list(allch_coords[row['peak_channel'],:]))
                unit_ipsi.append(is_ipsi[row['peak_channel']])
            unit_metrics['area'] = unit_area
            unit_metrics['ccf_coord'] = unit_coords
            unit_metrics['is_ipsi'] = unit_ipsi

            unit_metrics.to_csv(unit_info_location, index=False)
            print(' Saved {}.'.format(unit_info_location))


################################################################################
################################################################################

### Get location of stim electrode and save .json file ###

if stim_elec_location:
    print('\nGetting stim electrode location...')
    for expi, dataloci in data_location.items():
        if dataloci:
            print('\nExperiment {:d}: {}'.format(expi, os.path.basename(dataloci)))

            probe_annot_df = pd.read_csv(os.path.join(histology_loc, 'probe_annotations.csv'))
            init_coords_df = pd.read_csv(os.path.join(histology_loc, 'initial_ccf_coordinates_exp{:d}.csv'.format(expi)))

            stimdf = init_coords_df[init_coords_df['probe'] == 'probe' + stim_elec_location].reset_index(drop=True)
            try:
                stim_tip_DV = np.sort(
                    probe_annot_df['DV'][probe_annot_df['probe_name'] == 'Probe ' + stim_elec_location + str(expi)].values)[-1]
            except IndexError:
                print(' Stim electrode location was not annotated for this experiment.')
                continue

            surface_ind = np.nonzero(stimdf['structure_id'].values)[0][0]
            surface_region = structure_tree[structure_tree.id == stimdf['structure_id'].iloc[surface_ind]]['acronym'].iloc[0]
            surface_coords = np.array([
                stimdf['A/P'].iloc[surface_ind], stimdf['D/V'].iloc[surface_ind], stimdf['M/L'].iloc[surface_ind]])
            print(' Stim electrode surface: {}, [{:d},{:d},{:d}] (A/P,D/V,M/L)'.format(
                surface_region, surface_coords[0], surface_coords[1], surface_coords[2]
            ))

            tip_ind = stimdf.index[stimdf['D/V'] == stim_tip_DV].tolist()[0]
            tip_region = structure_tree[structure_tree.id == stimdf['structure_id'].iloc[tip_ind]]['acronym'].iloc[0]
            tip_coords = np.array([
                stimdf['A/P'].iloc[tip_ind], stimdf['D/V'].iloc[tip_ind], stimdf['M/L'].iloc[tip_ind]])
            print(' Stim electrode tip: {}, [{:d},{:d},{:d}] (A/P,D/V,M/L)'.format(
                tip_region, tip_coords[0], tip_coords[1], tip_coords[2]
            ))

            len_stim_elec = np.linalg.norm(surface_coords - tip_coords) * 0.025 # CCF: 25 um resolution
            print(' Length of stim electrode in brain: {:.2f} mm'.format(len_stim_elec))

            ## Save stim_elec_location.json ##
            stim_loc = {
                'surface': {'area': surface_region, 'ccf_coords': surface_coords.tolist()},
                'tip': {'area': tip_region, 'ccf_coords': tip_coords.tolist()},
            }
            filename = os.path.join(dataloci, 'experiment1', 'recording1', 'stim_elec_location.json')
            with open(filename, 'w') as outfile:
                json.dump(stim_loc, outfile, indent = 4, separators = (',', ': '))
            print(' Saved {}.'.format(filename))

        print('')
