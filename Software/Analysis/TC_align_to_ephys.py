"""

This script takes the output from the TC_annotation_app.py (probe_points.csv)
and creates the file used by the refinement step (initial_probe_coordinates.csv).
It creates the file by fitting a straight line to the probe points and
extrapolating that in the dorsal and ventral directions. It also extracts some
ephys metrics (LFP power in 0-10 Hz band, unit density, and multi-unit spike
counts) along the length of the probe and saves the plots as .png files. All
files created are saved in the selected histology folder.
* This will overwrite any pre-existing initial_ccf_coordinates.csv and probe
ephys .png files. *

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

################################################################################
################################################################################

### Template file locations and global variables ###
### Shouldn't need to change these each time ###

probes = (
    'Probe A1', 'Probe B1', 'Probe C1', 'Probe D1', 'Probe E1', 'Probe F1',
    'Probe A2', 'Probe B2', 'Probe C2', 'Probe D2', 'Probe E2', 'Probe F2'
) ## these are what probes are named in the TC_annotation_app ##

labels = np.load(
    '//allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/TC_CCF_template_files/annotation_volume_25um_by_index.npy'
)
volume_template = np.load(
    '//allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/TC_CCF_template_files/template_volume_25um.npy'
) ## orientation is: AP, DV, ML ##
structure_tree = pd.read_csv(
    '//allen/programs/braintv/workgroups/tiny-blue-dot/zap-n-zip/TC_CCF_template_files/structure_tree_safe_2017.csv'
)

################################################################################
################################################################################

### Create initial_ccf_coordinates.csv file for each experiment ###

probe_annotations = pd.read_csv(os.path.join(histology_loc, 'probe_annotations.csv'), index_col = 0)

df_columns = ['probe','structure_id', 'A/P','D/V','M/L']
df = pd.DataFrame(columns = df_columns)

for probe_idx, probe in enumerate(probes):
    x = probe_annotations[probe_annotations.probe_name == probe].ML
    y = probe_annotations[probe_annotations.probe_name == probe].DV
    z = probe_annotations[probe_annotations.probe_name == probe].AP

    if len(z) > 0:
        print(probe)
        data = np.vstack((z,y,x)).T ## this is now ordered AP, DV, ML
        datamean = data.mean(axis=0)
        D = data - datamean
        m1 = np.min(D[:,1]) * 2
        m2 = np.max(D[:,1]) * 2
        ## Find best fit line ##
        uu,dd,vv = np.linalg.svd(D)
        ## Extrapolate it above and below ##
        linepts = vv[0] * np.mgrid[-200:200:0.7][:,np.newaxis]
        linepts += datamean

        if linepts[-1,1] - linepts[0,1] < 0:
            linepts = np.flipud(linepts)

        structure_ids = np.zeros((linepts.shape[0],))
        ccf_coordinates = np.zeros((linepts.shape[0],3))
        for j in range(linepts.shape[0]):
            ccf_coordinate = np.around(linepts[j,:]).astype('int')
            ccf_coordinates[j,:] = ccf_coordinate
            if ccf_coordinate[1] < 0: # if D/V is neg, set structure_ids to -1
                structure_ids[j] = 0
            else:
                try:
                    structure_ids[j] = labels[ccf_coordinate[0], ccf_coordinate[1], ccf_coordinate[2]]
                except IndexError:
                    structure_ids[j] = -1

        data = {
            'probe': [probe]*linepts.shape[0],
            'structure_id': structure_ids.astype('int'),
            'A/P' : ccf_coordinates[:,0].astype('int'),
            'D/V' : ccf_coordinates[:,1].astype('int'),
            'M/L' : ccf_coordinates[:,2].astype('int')
        }
        probe_df = pd.DataFrame(data)
        df = pd.concat((df, probe_df), ignore_index=True)

exp_days = np.unique([x[-1] for x in df.probe.unique()]) # this will be 1 and/or 2 if there are multiple experiments annotated
for daynum in exp_days:
    mask = df['probe'].str.contains(daynum)
    split_df = df[mask].reset_index(drop=True)
    ## correct the probe name (removes capitals, space, and number) ##
    split_df['probe'] = split_df['probe'].apply(lambda x: x[0].lower() + x[1:-3] + x[-2])
    ## save each experiment as a separate file ##
    output_file = os.path.join(histology_loc, 'initial_ccf_coordinates_exp{}.csv'.format(daynum))
    print('Saving initial coordinates for experiment {} here: {}'.format(daynum, output_file))
    split_df.to_csv(output_file, index=False) # , index=False to avoid saving the indices

################################################################################
################################################################################

### Create the physiology plots for each probe and experiment ###

for expnum, dloc in data_location.items():
    if dloc is None:
        print('No data location set for experiment {:d}, not processing.'.format(expnum))
        continue
    all_probe_folders = glob(dloc + '/**/*_sorted', recursive=True)
    print('Found {:d} probes for experiment {:d}, processing...'.format(len(all_probe_folders), expnum))

    ## Loop through probes to extract ephys markers ##
    for probe_folder in all_probe_folders:
        probe_name = probe_folder[probe_folder.find('probe'):probe_folder.find('probe')+6]
        print(' ...{}'.format(probe_name))
        phys_output_file = os.path.join(histology_loc, 'physiology_{}_exp{:d}.png'.format(probe_name, expnum))


        ### Get surface ch estimate from probe_info.json ###
        probe_info_location = probe_folder + '/probe_info.json'
        with open(probe_info_location) as data_file:
            data = json.load(data_file)
        surface_ch = int(data['surface_channel'])


        ### LFP power ###
        LFP_cont_location = probe_folder + '/continuous/Neuropix-PXI-100.1/continuous.dat'
        raw_data = np.memmap(LFP_cont_location, dtype='int16')
        data = np.reshape(raw_data, (int(raw_data.size / 384), 384))

        ## Get power in 10 s window, beginning 60 s after recording start ##
        start_index = int(2500 * 60) # start 60 s into rec
        end_index = start_index + 25000 # look at next 10 s

        ## Design Butterworth filter: 1-1000 Hz ##
        b,a = butter(3,[1/(2500/2),1000/(2500/2)],btype='band')

        ## Convert the data to voltage ##
        D = data[start_index:end_index,:] * 0.195

        ## Apply the bandpass filter ##
        for i in range(D.shape[1]):
            D[:,i] = filtfilt(b,a,D[:,i])

        ## Subtract the median of top 14 chs ##
        M = np.median(D[:,370:])
        for i in range(D.shape[1]):
            D[:,i] = D[:,i] - M

        ## Get the power across the chs ##
        nfft = 2048
        power = np.zeros((int(nfft/2+1), 384))
        for channel in range(D.shape[1]):
            sample_frequencies, Pxx_den = welch(D[:,channel], fs=2500, nfft=nfft)
            power[:,channel] = Pxx_den
        in_range = (sample_frequencies > 0) * (sample_frequencies < 10)


        ### Multi-unit firing rate ###
        spthresh = 50
        AP_cont_location = probe_folder + '/continuous/Neuropix-PXI-100.0/continuous.dat'
        raw_apdata = np.memmap(AP_cont_location, dtype='int16')
        apdata = np.reshape(raw_apdata, (int(raw_apdata.size / 384), 384))
        ## Choose the time window to analyze: 60 s after beginning, for 3 s ##
        start_index = int(30000 * 60)
        end_index = start_index + (30000 * 3)
        ## Bandpass filter: 0.5-10 kHz
        b, a = butter(3, [500/(30000/2), 10000/(30000/2)], btype='band')
        ## Count threshold crossings ##
        thresh_crossings = np.zeros((384), dtype=int)
        for chi in range(384):
            chdata = apdata[start_index:end_index, chi] * 0.195
            chdata = filtfilt(b, a, chdata)
            spikes = find_peaks(-chdata, height=spthresh)[0]
            thresh_crossings[chi] = len(spikes)
        ## Smooth the metric a little ##
        spcount = gaussian_filter1d(thresh_crossings, 0.8)


        ### Unit density ###
        cluster_group = pd.read_csv(probe_folder + '/continuous/Neuropix-PXI-100.0/cluster_group.tsv', sep='\t')
        cluster_metrics = pd.read_csv(probe_folder + '/continuous/Neuropix-PXI-100.0/metrics.csv')
        spike_clusters = np.load(probe_folder + '/continuous/Neuropix-PXI-100.0/spike_clusters.npy')

        ## Select good units ##
        isi_viol_thresh=0.5
        amp_cutoff_thresh=0.1
        if np.array_equal(cluster_group['cluster_id'].values.astype('int'), cluster_metrics['cluster_id'].values.astype('int')):
            unit_metrics = pd.merge(
                cluster_group.rename(columns={'group':'label'}),
                cluster_metrics, #.drop(cluster_metrics.columns[0], axis=1), this drop causes a problem
                on='cluster_id'
            )
        else:
            print('  IDs from cluster_group DO NOT match cluster_metrics, not creating physiology plot .png.')
            continue
        selected_metrics = unit_metrics[(
            (unit_metrics['label'] == 'good') &
            (unit_metrics['isi_viol'] < isi_viol_thresh) &
            (unit_metrics['amplitude_cutoff'] < amp_cutoff_thresh)
        )]
        units = selected_metrics.cluster_id.values
        unit_channels = selected_metrics.peak_channel.values

        ## Get histogram of units per channel ##
        unit_histogram = np.zeros((384), dtype='float')
        total_units = 0
        for unit_idx, unit in enumerate(units):
            channel = unit_channels[unit_idx]
            unit_histogram[channel] += 1
            total_units += 1
        GF = gaussian_filter1d(unit_histogram*500, 2)


        ### Create plot and save ###
        fig = plt.figure(frameon=False)
        plt.clf()
        fig.set_size_inches(1,8)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ## LFP mean power ##
        ax.plot(np.mean(power[in_range,:],0), np.arange(384), '.', color='pink')
        ax.axhline(surface_ch, color='g', alpha=0.5)
        ## Multi-unit spike count ##
        muax = ax.twiny()
        muax.plot(spcount, np.arange(384), linewidth=2.0, alpha=0.6, color='purple')
        ## Unit histogram ##
        uax = ax.twiny()
        uax.barh(np.arange(384), GF, height=1.0, alpha=0.1, color='teal')
        uax.plot(GF, np.arange(384), linewidth=2.0, alpha=0.6, color='teal')
        plt.ylim([0, 384])
        # plt.xlim([-5, 2000])
        fig.savefig(phys_output_file, dpi=300)
