import numpy as np
import math
import matplotlib.pyplot as plt
from so3g.hk import load_range
from sotodlib.tod_ops.flags import get_glitch_flags, get_trending_flags
from sotodlib.tod_ops.detrend import detrend_tod
import latrt_testing.fft_ops as fft_ops
import latrt_testing.demodulation as demod
from tqdm import tqdm
import os
from sotodlib.io import hkdb

def find_time(timestamps, time):
    '''finds closest time index in array timestamps to the inputted time'''
    return (np.abs(timestamps - time)).argmin()


# useful function for analyzing timestreams
def time_zoom(aman, t_min, t_max):
    # returns indices useful for getting a time window
    time = aman.timestamps - aman.timestamps[0]
    inds = np.where((time >= t_min) & (time <= t_max))[0]
    return inds

def get_middle_psd(aman, middle_ind=None, window_length=12000,
                   nperseg=2**8):
    """Get PSD using a chunk of aman in the middle."""
    if middle_ind is None:
        middle_ind = aman.timestamps.shape[0] // 2
    psd_aman = aman.restrict('samps', (middle_ind - window_length // 2,
                                       middle_ind + window_length // 2),
                             in_place=False)
    Pxx, freqs = fft_ops.psd(psd_aman, nperseg=nperseg)
    return Pxx, freqs


def get_good_dets(aman, Pxx, freqs, power_threshold=1000, plot=False,
                  chop_freq=8, apply_trend_cuts=True):

    """Take out trends"""
    # get the 'good' dets with lots of power at 8Hz
    if (plot):
        plt.figure()

    good_dets = []
    if apply_trend_cuts:
        trend_cuts = aman.flags.has_cuts(['trends'])
    for i in range(0, aman.dets.count):
        if apply_trend_cuts:
            if aman.dets.vals[i] in trend_cuts:
                continue

        index_of_8hz = np.where(freqs <= chop_freq)[0][-1]
        if Pxx[i, index_of_8hz] > (Pxx[i, index_of_8hz + 10] * (
                power_threshold)):
            good_dets.append(i)
            if (plot):
                plt.loglog(freqs, Pxx[i])

    if (plot):
        plt.xlim(chop_freq - 2, chop_freq + 2)
        plt.grid()
        plt.xlabel('frequency (hz)')
        plt.ylabel('P(f)')
        plt.title('power spectra of detectors with good 8hz power')
        plt.show()

    good_dets = np.array(good_dets)

    return good_dets


def get_fts_ind_ranges(fts_position_inds):
    time_interval = 200
    ind_ranges = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if len(inds) > 2:
            ind_start, ind_end = inds[0], inds[-2]
        # If there's only one housekeeping index (happens rarely with >1s
        # integration unless it skips a data point), get the previous one
        # which is further away and integrate in that direction.
        if ind_start == ind_end:
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        ind_ranges.append(ind_range)
    return ind_ranges


def get_integration_indices_optimized(fts_ind_ranges, glitch_mask):
    total_non_glitch_inds = []
    for ind_range in fts_ind_ranges:
        mask = glitch_mask[np.where((glitch_mask >= ind_range[0]) & (
            glitch_mask <= ind_range[-1]))]
        non_glitch_inds = np.setdiff1d(ind_range, mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def get_integration_indices(fts_position_inds, glitch_mask):
    # So we just need to integrate between all of our times and discount the
    # glitches
    # First we need to make sure that the glitches exist
    time_interval = 200
    total_non_glitch_inds = []
    for i, inds in enumerate(fts_position_inds):
        ind_start, ind_end = inds[0], inds[-1]
        if ind_start == ind_end:
            # get the previous one which is further away and integrate in that
            # direction.
            next_ind = fts_position_inds[i + 1][0]
            prev_ind = fts_position_inds[i - 1][-1]
            if (next_ind - ind_start) > (ind_start - prev_ind):
                ind_end = int(ind_start + time_interval / 2)
            else:
                ind_start = int(ind_end - time_interval / 2)

        # Integrate between these
        ind_range = np.arange(ind_start, ind_end + 1)
        non_glitch_inds = np.setdiff1d(ind_range, glitch_mask)
        total_non_glitch_inds.append(non_glitch_inds)
    return total_non_glitch_inds


def integrate_signal(signal, total_non_glitch_inds):
    return np.array([np.mean(signal[inds]) for inds in total_non_glitch_inds])


def load_fts_range(aman, resolution=.15):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')

    max_position = -1 * np.round(np.min(hk_data['fts_mirror'][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position,
        int(2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data['fts_mirror'][1]
    # now take out the initial data chunk
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[last_max_index:]
    hk_times = hk_data['fts_mirror'][0][last_max_index:]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_index = hk_inds[0]
            hk_position = hk_mirror_positions[hk_index]
            hk_time = hk_times[hk_index]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    #assert (np.abs(hk_mirror_slice - expected_fts_mirror_positions) <= .01).all()

    aman_fts_position_timeslice = np.array(
        [find_time(aman.timestamps, time) for time in hk_time_slice])
    return aman_fts_position_timeslice, np.array(hk_mirror_slice)


def load_fts_range_bounds(aman, resolution=.15, max_position=None):
    # hk_data = load_range(
    #     float(aman.timestamps[0]), float(aman.timestamps[-1]),
    #     config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')

    # hk_data = load_range(
    #     float(aman.timestamps[0]), float(aman.timestamps[-1]),
    #     data_dir="/so/level2-daq/lat/hk",
    #     lat = ['fields.fts-uchicago-act.feeds.position.pos'])

    cfg = hkdb.HkConfig.from_yaml('/global/cfs/cdirs/sobs/users/mhasse/work/250404/hkdb-lat.cfg')
    lspec = hkdb.LoadSpec(
        cfg=cfg, start=aman.timestamps[0], end=aman.timestamps[-1],
        fields = ["fts-uchicago-act.position.*"],
    )

    result = hkdb.load_hk(lspec, show_pb=True)
    hk_times, hk_mirror_positions = result.data['fts-uchicago-act.position.pos']
    start_ind = np.where(hk_times > aman.timestamps[0])[0][0]
    end_ind = np.where(hk_times > aman.timestamps[-1])[0][0]
    hk_times = hk_times[start_ind : end_ind]
    hk_mirror_positions = hk_mirror_positions[start_ind : end_ind]
    # print(hk_mirror_positions)
    # return hk_mirror_positions, hk_times

    if max_position is None:
        max_position = -1 * np.round(np.min(hk_mirror_positions), 2)
    print(max_position)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position, int(
            2 * max_position / resolution) + 1), 6)
    # now take out the initial data chunk
    # start slightly after the beginning to account for any weird trends
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][2]
    # start slightly before the end similarly
    first_right_max_index = np.where(
        np.abs(hk_mirror_positions - max_position) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[
        last_max_index: first_right_max_index]
    hk_times = hk_times[last_max_index: first_right_max_index]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= .01)[0]
        if len(hk_inds) == 0:
            print(f"no housekeeping data at fts position {pos}. "
                  "Using data from previous position")
            hk_position = hk_mirror_slice[-1]
            hk_time = hk_time_slice[-1]
        else:
            hk_position = hk_mirror_positions[hk_inds][0]
            hk_time = hk_times[hk_inds]
        hk_mirror_slice.append(hk_position)
        hk_time_slice.append(hk_time)

    aman_fts_position_timeslice = [
        [find_time(aman.timestamps, time) for time in s] for s in (
            hk_time_slice)]
    return hk_mirror_slice, aman_fts_position_timeslice


def plot_good_interferograms(aman, good_dets, signal, fts_mirror_positions,
                             figsize=(10, 10)):
    n_bias_groups = np.max(aman.det_info.smurf.bias_group) + 1
    fig, axes = plt.subplots(math.ceil(n_bias_groups / 2), 2, figsize=figsize)
    axes = axes.ravel()

    for group in range(n_bias_groups):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.bias_group[good_dets] == group)
        axes[group].set_title(
            "bias group %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in bias group -1: %s' % np.sum(
        aman.det_info.smurf.bias_group[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in aman.flags.has_cuts(['trends']) or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.bias_group[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def plot_good_interferograms_bands(aman, good_dets, signal, fts_mirror_positions,
                                   figsize=(10, 10)):
    n_bands = np.max(aman.det_info.smurf.band) + 1
    fig, axes = plt.subplots(math.ceil(n_bands / 2), 2, figsize=figsize)
    axes = axes.ravel()
    trend_cuts = aman.flags.has_cuts(['trends'])

    for group in range(n_bands):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.band[good_dets] == group)
        axes[group].set_title(
            "band %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in band -1: %s' % np.sum(
        aman.det_info.smurf.band[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in trend_cuts or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.band[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def save_data(aman, n, fts_mirror_positions, signal, folder_name,
              band_channel_map, obs_id):
    # fts_x, fts_y = get_fts_position(aman)

    # save the data for loading in from another notebook
    trend_cuts = aman.flags.has_cuts(['trends'])
    data = np.zeros((len(fts_mirror_positions), len(band_channel_map)))
    bands = np.zeros(len(band_channel_map))
    channels = np.zeros(len(band_channel_map))
    det_ids = np.empty(len(band_channel_map), dtype="<U18")
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
        det_id = aman.det_info.det_id[i]
        # print(band, channel)
        if aman.dets.vals[i] in trend_cuts:
            # just make this data a bunch of zeros
            # adjust this to actually save data-- don't trust trends lol
            data[:, band_channel_id] = signal[i]
            # data[:, band_channel_id] = np.zeros(len(fts_mirror_positions))
        else:
            data[:, band_channel_id] = signal[i]

        bands[band_channel_id] = band
        channels[band_channel_id] = channel
        det_ids[band_channel_id] = det_id

    filename = '%s/run_%s_interferograms.npz' % (folder_name, n)
    with open(filename, 'wb') as f:
        np.savez(f, data=data, #xy_position=(fts_x, fts_y),
                 fts_mirror_positions=fts_mirror_positions,
                 bands=bands, channels=channels, det_ids=det_ids,
                 obs_id=obs_id)
    print('data saved to location %s' % filename)
    return


def get_fts_position(aman):
    # get the XY stage position
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')
    assert np.around(np.std(hk_data['xy_stage_x'][1]), 2) == 0
    assert np.around(np.std(hk_data['xy_stage_y'][1]), 2) == 0

    fts_x, fts_y = np.around(np.mean(hk_data['xy_stage_x'][1]), 1), np.round(
        np.mean(hk_data['xy_stage_y'][1]), 1)
    return fts_x, fts_y



def check_chopper_signal(aman, power_threshold=100, chop_freq=8,
                         nperseg=2**10, return_good_aman=False):
    print(f"Length of chop = {aman.timestamps[-1] - aman.timestamps[0]}")
    get_trending_flags(aman, t_piece=50)
    detrend_tod(aman)
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)
    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    good_dets = get_good_dets(aman, Pxx, freqs, plot=True,
                              power_threshold=power_threshold,
                              chop_freq=chop_freq, apply_trend_cuts=False)
    print(f"Number of good dets = {len(good_dets)}")
    plt.plot(freqs, Pxx[good_dets].T, alpha=0.1)
    plt.yscale('log')
    # plt.axvline(13, ls='--', color="black")
    # plt.axvline(2.8, ls='--', color="black")
    plt.axvline(8, ls='--', color="black")
    plt.xlim(0, 22)
    plt.ylim(1e-8, 1e-1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.show()

    phase_fit_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False)
    phase_to_use, phases = demod.fit_phase(phase_fit_aman, 30, plot=True,
                                           threshold=0.2, index_limit=180,
                                           freq=chop_freq)
    if np.std(phases) > .3:
        print('Phase fitting standard deviation is slightly high, check hist')
        plt.hist(phases)
        plt.xlabel('phase')
        plt.ylabel('counts')
        plt.grid()
        plt.show()

    demod.demod_single_sine(phase_fit_aman, phase_to_use, lp_fc=0.5,
                            freq=chop_freq)

    plt.plot(phase_fit_aman.timestamps - phase_fit_aman.timestamps[0],
             phase_fit_aman.demod_signal.T, alpha=0.3)
    [plt.axhline(m, alpha=0.1, ls="--", color=f"C{i}") for i, m in enumerate(
        np.median(phase_fit_aman.demod_signal, axis=1))]
    #plt.ylim(-0.02, 0.4)
    #plt.xlim(20, 20 + (5 / 32))
    plt.show()
    if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
        print_num_in_each_band(phase_fit_aman)
    if not return_good_aman:
        return
    return phase_fit_aman


def print_num_in_each_band(aman):
    assert 'bias_group' in aman.det_info.smurf.keys()
    bg_mapping = {0: '90', 1: '90', 2: '150', 3: '150', 4: '90', 5: '90',
                  6: '150', 7: '150', 8: '90', 9: '90', 10: '150', 11: '150'}
    bgs, counts = np.unique(aman.det_info.smurf.bias_group,
                            return_counts=True)
    total_90s = 0
    total_150s = 0
    for bg, count in zip(bgs, counts):
        if bg == -1:
            continue
        if bg_mapping[bg] == '90':
            total_90s += count
        else:
            total_150s += count
    print(f"total 90s = {total_90s}")
    print(f"total 150s = {total_150s}")
    return


def process_run_ufm(aman, folder_name, band_channel_map,
                    middle_relative_time=2000, threshold=.1, index_limit=160,
                    plot=False, resolution=.1, nperseg=(2 ** 9),
                    demod_lp_fc=0.5, chop_freq=8, max_position=None,
                    run_num=0):
    assert os.path.exists(folder_name)
    get_trending_flags(aman)
    detrend_tod(aman)

    # get the glitches.
    _ = get_glitch_flags(aman, hp_fc=1.0, buffer=20, overwrite=True, n_sig=50)
    mask = aman.flags.glitches.mask()

    # get the 'good' dets with lots of power at 8Hz
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)
    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    for power_threshold in [100, 10]:
        print('using power threshold of %s:' %power_threshold)
        good_dets = get_good_dets(aman, Pxx, freqs, plot=plot,
                                  power_threshold=power_threshold,
                                  chop_freq=chop_freq)
        if len(good_dets) > 80:
            break
    print('number of detectors with higher power in 8hz = %s' %len(good_dets))
    if len(good_dets) > 10:
        # now fit the phase with the 'good' detectors
        phase_fit_aman = aman.restrict('dets', aman.dets.vals[good_dets], in_place=False)
        phase_to_use, phases = demod.fit_phase(phase_fit_aman, middle_relative_time, plot=plot,
                                               threshold=threshold, index_limit=index_limit,
                                               freq=chop_freq)
        if np.std(phases) > .3:
            print('Phase fitting standard deviation is slightly high, check hist')
            plt.hist(phases)
            plt.xlabel('phase')
            plt.ylabel('counts')
            plt.grid()
            plt.show()

        if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
            print_num_in_each_band(phase_fit_aman)

        # demodulate with the fitted phase
        demod.demod_single_sine(aman, phase_to_use, lp_fc=demod_lp_fc, freq=chop_freq)
    else:
        print('not enough good detectors found to fit phase. demodulating with a sine + cosine')
        demod.demod_sine(aman, freq=chop_freq, lp_fc=demod_lp_fc)

    print("getting integrated signal...")
    # get the integrated signal
    fts_mirror_positions, fts_time_ranges  = load_fts_range_bounds(
        aman, resolution=resolution, max_position=max_position)
    interferograms = []
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    for i in tqdm(range(len(mask))):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(
            fts_ind_ranges, np.where(mask[i])[0])
        integrated_signal = integrate_signal(
            aman.demod_signal[i], total_non_glitch_inds)
        interferograms.append(integrated_signal)
    interferograms = np.array(interferograms)

    if plot:
        if len(good_dets) > 0:
            if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
                plot_good_interferograms(aman, good_dets, interferograms,
                                        fts_mirror_positions)
            else:
                plot_good_interferograms_bands(aman, good_dets, interferograms,
                                            fts_mirror_positions)


        else:
            if 'bias_group' in phase_fit_aman.det_info.smurf.keys():
                plot_good_interferograms(aman, list(range(aman.dets.count)),
                                        interferograms, fts_mirror_positions)

            else:
                plot_good_interferograms_bands(aman, list(range(aman.dets.count)),
                                            interferograms, fts_mirror_positions)


    # save this data along with bias group number, dets, and XY position to another notebook
    save_data(aman, run_num, fts_mirror_positions, interferograms,
              folder_name, band_channel_map, int(aman.timestamps[0]))
    return interferograms
