import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from so3g.hk import load_range
from sotodlib.tod_ops.flags import get_glitch_flags, get_trending_flags
from sotodlib.core import AxisManager
import latrt_testing.fft_ops as fft_ops
import os
import latrt_testing.demodulation as demod
from sotodlib.tod_ops.detrend import detrend_tod
from sotodlib.io import load_smurf

#HK_KEY = 'lat.fts-uchicago-act.feeds.position.pos'
#HK_KEY = 'lat.fts-uchicago-act.feeds.position.pos'
HK_KEY = 'satp1.fts-uchicago-so.feeds.position.pos'
HK_DATA_DIR = "/so/level2-daq/satp1/hk"

test_var = 8


def load_l2_file(ctime, streamid, l2path='/so/level2-daq/lat/timestreams'):
    bookctime = str(ctime)
    ufm_path = os.path.join(l2path,bookctime[:5],streamid)
    flist = []
    # sort timestamps if you want the data to be in order!!!
    for fp in np.sort(os.listdir(ufm_path)):
        if abs(int(fp.split('_')[0]) - int(bookctime)) < 30:
            flist.append(os.path.join(ufm_path, fp))
    return load_smurf.load_file(flist)

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


def get_good_dets(aman, Pxx, freqs, power_threshold=100, plot=False,
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


def integrate_signal(signal, total_non_glitch_inds, integration_function="median"):
    assert integration_function in ["mean", "median"]
    int_funcs = {"mean": np.mean, "median": np.median}
    func_to_use = int_funcs[integration_function]
    return np.array([func_to_use(signal[inds]) for inds in total_non_glitch_inds])


def load_fts_range(aman, resolution=.15):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        data_dir=HK_DATA_DIR,
        fields = [HK_KEY])

    max_position = -1 * np.round(np.min(hk_data[HK_KEY][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position,
        int(2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data[HK_KEY][1]
    # now take out the initial data chunk
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= .01)[0][-2]
    hk_mirror_positions = hk_mirror_positions[last_max_index:]
    hk_times = hk_data[HK_KEY][0][last_max_index:]
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


def load_fts_range_bounds(aman, resolution=.15, max_position=None, error=0.01):
    hk_data = load_range(
        float(aman.timestamps[0]), float(aman.timestamps[-1]),
        data_dir=HK_DATA_DIR,
        fields = [HK_KEY])

    if max_position is None:
        max_position = -1 * np.round(np.min(hk_data[HK_KEY][1]), 2)
    expected_fts_mirror_positions = np.round(np.linspace(
        -1 * max_position, max_position, int(
            2 * max_position / resolution) + 1), 6)
    hk_mirror_positions = hk_data[HK_KEY][1]
    # now take out the initial data chunk
    # start slightly after the beginning to account for any weird trends
    last_max_index = np.where(
        np.abs(hk_mirror_positions - (-1 * max_position)) <= error)[0][2]
    # start slightly before the end similarly
    first_right_max_index = np.where(
        np.abs(hk_mirror_positions - max_position) <= error)[0][-2]
    hk_mirror_positions = hk_mirror_positions[
        last_max_index: first_right_max_index]
    hk_times = hk_data[HK_KEY][0][last_max_index: first_right_max_index]
    hk_mirror_slice = []
    hk_time_slice = []
    for pos in expected_fts_mirror_positions:
        hk_inds = np.where(np.abs(hk_mirror_positions - pos) <= error)[0]
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
    trend_cuts = aman.flags.has_cuts(['trends'])

    for group in range(n_bias_groups):
        axes[group].grid(True)
        count = np.sum(aman.det_info.smurf.bias_group[good_dets] == group)
        axes[group].set_title(
            "bias group %s, number of 'good' dets = %s" % (group, count))

    print('number of interferograms in bias group -1: %s' % np.sum(
        aman.det_info.smurf.bias_group[good_dets] == -1))

    for i in range(0, aman.dets.count):
        if aman.dets.vals[i] in trend_cuts or np.max(
                signal[i]) > 1:
            continue

        group = aman.det_info.smurf.bias_group[i]
        if (group != -1) and i in good_dets:
            axes[group].plot(fts_mirror_positions, signal[i])
    plt.tight_layout()
    plt.show()


def plot_good_interferograms_bands(aman, good_dets, signal, fts_mirror_positions,
                                   figsize=(10, 10)):
    n_bands= np.max(aman.det_info.smurf.band) + 1
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
    for i in range(aman.dets.count):
        band, channel = aman.det_info.smurf.band[i], aman.det_info.smurf.channel[i]
        band_channel_id = band_channel_map[(band, channel)]
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

    filename = '%s/run_%s_interferograms.npz' % (folder_name, n)
    with open(filename, 'wb') as f:
        np.savez(f, data=data, #xy_position=(fts_x, fts_y),
                 fts_mirror_positions=fts_mirror_positions,
                 bands=bands, channels=channels, obs_id=obs_id)
    print('data saved to location %s' % filename)
    return


# def get_fts_position(aman):
#     # get the XY stage position
#     hk_data = load_range(
#         float(aman.timestamps[0]), float(aman.timestamps[-1]),
#         config='/data/users/kmharrin/smurf_context/hk_config_202104.yaml')
#     assert np.around(np.std(hk_data['xy_stage_x'][1]), 2) == 0
#     assert np.around(np.std(hk_data['xy_stage_y'][1]), 2) == 0

#     fts_x, fts_y = np.around(np.mean(hk_data['xy_stage_x'][1]), 1), np.round(
#         np.mean(hk_data['xy_stage_y'][1]), 1)
#     return fts_x, fts_y

def process_run_ufm(aman, folder_name, band_channel_map,
                    middle_relative_time=2000, threshold=.1, index_limit=160,
                    plot=False, resolution=.1, nperseg=(2 ** 9),
                    demod_lp_fc=0.5, chop_freq=8, max_position=None,
                    run_num=0, hk_error=0.01):
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

    # get the integrated signal
    fts_mirror_positions, fts_time_ranges = load_fts_range_bounds(
        aman, resolution=resolution, max_position=max_position, error=hk_error)
    interferograms = []
    fts_ind_ranges = get_fts_ind_ranges(fts_time_ranges)
    for i in range(len(mask)):
        # integrate around any glitches in the data.
        total_non_glitch_inds = get_integration_indices_optimized(
            fts_ind_ranges, np.where(mask[i])[0])
        integrated_signal = integrate_signal(
            aman.demod_signal[i], total_non_glitch_inds)
        interferograms.append(integrated_signal)
    interferograms = np.array(interferograms)

    if (plot and len(good_dets) > 0):
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
    return

def check_chopper_signal(aman, power_threshold=100, chop_freq=8,
                         nperseg=2**10, t_piece=20, return_good_aman=False):
    print(f"Length of chop = {aman.timestamps[-1] - aman.timestamps[0]}")
    get_trending_flags(aman, t_piece=t_piece)
    detrend_tod(aman)
    # Pxx, freqs = fft_ops.psd(aman, nperseg=nperseg)
    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    good_dets = get_good_dets(aman, Pxx, freqs, plot=True,
                              power_threshold=power_threshold,
                              chop_freq=chop_freq, apply_trend_cuts=False)
    print(f"Number of good dets = {len(good_dets)}")
    plt.plot(freqs, Pxx[good_dets].T, alpha=0.1)
    plt.yscale('log')
    #plt.axvline(13, ls='--', color="black")
    #plt.axvline(2.8, ls='--', color="black")
    plt.axvline(chop_freq, ls='--', color="black")
    plt.xlim(0, 2 * chop_freq)
    plt.ylim(1e-8, 1e-3)
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

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    return offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def fit_centroid(x, y, weights, verbose=True):
    # Initial guesses from SNR-weighted centroid
    x_cen0 = np.average(x, weights=weights)
    y_cen0 = np.average(y, weights=weights)
    spread_x = np.sqrt(np.average((x - x_cen0)**2, weights=weights))
    spread_y = np.sqrt(np.average((y - y_cen0)**2, weights=weights))
    p0 = [np.max(weights), x_cen0, y_cen0, spread_x, spread_y, 0.0, np.min(weights)]

    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), weights, p0=p0,
                            maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        x_centroid = popt[1]
        y_centroid = popt[2]
        if verbose:
            print(f'2D Gaussian fit parameters:')
            print(f'  Amplitude = {popt[0]:.1f} ± {perr[0]:.1f}')
            print(f'  x_peak   = {popt[1]:.6f} ± {perr[1]:.6f}')
            print(f'  y_peak  = {popt[2]:.6f} ± {perr[2]:.6f}')
            print(f'  sigma_x  = {popt[3]:.6f} ± {perr[3]:.6f}')
            print(f'  sigma_y = {popt[4]:.6f} ± {perr[4]:.6f}')
            print(f'  theta     = {np.degrees(popt[5]):.1f}° ± {np.degrees(perr[5]):.1f}°')
            print(f'  offset    = {popt[6]:.1f} ± {perr[6]:.1f}')
        centroid_label = '2D Gaussian peak'
    except RuntimeError as e:
        print(f'Gaussian fit failed: {e}')
        print('Falling back to SNR-weighted centroid')
        x_centroid = x_cen0
        y_centroid = y_cen0
        popt = None
        centroid_label = 'SNR-weighted centroid (fallback)'

    if verbose:
        print(f'\nCentroid: x={x_centroid:.6f}, y={y_centroid:.6f}')

    return x_centroid, y_centroid, popt

def add_wafer_xy_angle(aman_signal, aman_map, verbose=True):
    """
    Add wafer.x, wafer.y, wafer.angle and det_cal.bg to aman_signal by matching
    (smurf.band, smurf.channel) against aman_map.

    Parameters
    ----------
    aman_signal : AxisManager
        AxisManager with timestreams (e.g. aman_wl_on)
    aman_map : AxisManager
        AxisManager containing wafer.x, wafer.y, and det_cal.bg
    verbose : bool
        Print matching summary

    Returns
    -------
    None
        Modifies aman_signal in place
    """

    # --- Build lookup from mapping aman ---
    map_lookup = {
        (int(b), int(c)): (float(x), float(y), float(angle), int(bg))
        for b, c, x, y, angle, bg in zip(
            aman_map.det_info.smurf.band,
            aman_map.det_info.smurf.channel,
            aman_map.det_info.wafer.x,
            aman_map.det_info.wafer.y,
            aman_map.det_info.wafer.angle,
            aman_map.det_cal.bg,
        )
    }

    n_det = aman_signal.dets.count

    x_new = np.full(n_det, np.nan)
    y_new = np.full(n_det, np.nan)
    angle_new = np.full(n_det, np.nan)
    bg_new = np.full(n_det, -1, dtype=int)  # -1 = unmatched

    # --- Match detectors ---
    for d in range(n_det):
        key = (
            int(aman_signal.det_info.smurf.band[d]),
            int(aman_signal.det_info.smurf.channel[d]),
        )

        if key in map_lookup:
            x_new[d], y_new[d], angle_new[d], bg_new[d] = map_lookup[key]

    # --- Attach wafer coords ---
    if hasattr(aman_signal.det_info, "wafer"): # Note level 2 data does have a 'det_info' field.
        aman_signal.det_info.wafer.x = x_new
        aman_signal.det_info.wafer.y = y_new
        aman_signal.det_info.wafer.angle = angle_new
    else:
        wafer = AxisManager(aman_signal.dets)
        wafer.wrap("x", x_new, [(0, "dets")])
        wafer.wrap("y", y_new, [(0, "dets")])
        wafer.wrap("angle", angle_new, [(0, "dets")])
        aman_signal.det_info.wrap("wafer", wafer) 

    # --- Attach bg ---
    if hasattr(aman_signal, "det_cal") and hasattr(aman_signal.det_cal, "bg"):
        aman_signal.det_cal.bg = bg_new
    else:
        if not hasattr(aman_signal, "det_cal"):
            det_cal = AxisManager(aman_signal.dets)
            aman_signal.wrap("det_cal", det_cal)

        aman_signal.det_cal.wrap("bg", bg_new, [(0, "dets")])

    # --- Sanity check ---
    if verbose:
        matched = bg_new >= 0
        print(f"Matched wafer coords + bg for {matched.sum()} / {n_det} detectors")

        if matched.any():
            d = np.where(matched)[0][0]
            print(
                "Example match:",
                f"band={aman_signal.det_info.smurf.band[d]},",
                f"chan={aman_signal.det_info.smurf.channel[d]},",
                f"x={aman_signal.det_info.wafer.x[d]:.3f},",
                f"y={aman_signal.det_info.wafer.y[d]:.3f},",
                f"angle={aman_signal.det_info.wafer.angle[d]:.1f},",
                f"bg={aman_signal.det_cal.bg[d]}",
            )

def check_pol_snr(aman, aman_map, chop_freq=8, nperseg=2**10, t_piece=20,
                  apply_trend_cuts=True, log_scale=False, pct_lo=2, pct_hi=98):
    '''
    Computes a rough SNR for detectors in a timestream (meant as a sanity check for a short chop).
    Fits a 2d Gaussian to the illuminated detectors and finds the centroid.
    aman: AxisManager with timestreams
    aman_map: AxisManager with wafer.x, wafer.y, wafer.angle and det_cal.bg

    Parameters
    ----------
    log_scale : bool
        If True, use logarithmic color normalization (good when SNR spans decades).
    pct_lo, pct_hi : float
        Lower and upper percentiles for robust color clipping (default 2nd–98th).

    Outputs:
    Plot of SNR vs polarization angle
    Polar plot of polarization angle (0-180 deg) vs radial distance from centroid, colored by SNR
    '''
    # add wafer coords to aman using aman_map
    add_wafer_xy_angle(aman, aman_map)

    # get snr for all detectors    
    print(f"Length of chop = {aman.timestamps[-1]-aman.timestamps[0]:.1f} seconds")
    get_trending_flags(aman, t_piece=t_piece)
    detrend_tod(aman)
    trend_mask = np.isin(aman.dets.vals, aman.flags.has_cuts(['trends']))
    print(f"Number of trending detectors: {np.sum(trend_mask)}")

    Pxx, freqs = get_middle_psd(aman, nperseg=nperseg)
    chop_freq_index = np.where(freqs <= chop_freq)[0][-1]
    snr = Pxx[:, chop_freq_index]/Pxx[:, chop_freq_index+10]

    # filter out NaNs and trending dets (if apply_trend_cuts=True)
    pol_angles = aman.det_info.wafer.angle
    valid = ~np.isnan(pol_angles) & ~np.isnan(snr)
    if apply_trend_cuts:
        valid &= ~trend_mask
    x = aman.det_info.wafer.x[valid]
    y = aman.det_info.wafer.y[valid]
    pol_angles = pol_angles[valid]
    snr = snr[valid]

    # fit centroid
    x_centroid, y_centroid, popt = fit_centroid(x, y, snr)

    # --- Build robust color normalization ---
    snr_finite = snr[np.isfinite(snr)]
    vmin = np.percentile(snr_finite, pct_lo)
    vmax = np.percentile(snr_finite, pct_hi)
    if log_scale:
        vmin = max(vmin, snr_finite[snr_finite > 0].min())  # avoid log(0)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # plot focal plane with SNR color and centroid
    if popt is not None:
        fig_fp, ax_fp = plt.subplots(figsize=(8, 7))
        sc = ax_fp.scatter(x, y, c=snr, cmap=cmap, norm=norm,
                   edgecolors='k', linewidths=0.5, alpha=0.8)
        plt.colorbar(sc, ax=ax_fp, label='SNR')
        ax_fp.scatter(x_centroid, y_centroid, color='r', marker='*', s=200, linewidths=3, 
              zorder=10, label='Fitted centroid')
        ax_fp.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.15, 1))
        ax_fp.set_xlabel('x')
        ax_fp.set_ylabel('y')
        ax_fp.set_title(f'Focal Plane: SNR Color Scale (clipped {pct_lo}–{pct_hi} pct)')
        ax_fp.set_aspect('equal')

        # Radial distance from centroid
        r_from_centroid = np.sqrt((x - x_centroid)**2 + (y - y_centroid)**2)

        # Polar plot: radial = distance from centroid, angle = folded pol angle, color = SNR
        fig_pol, ax_pol = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        # Normalize alpha to [0.05, 1.0] based on clipped SNR range
        snr_clipped = np.clip(snr, vmin, vmax)
        if vmax > vmin:
            alphas = 0.05 + 0.95 * (snr_clipped - vmin) / (vmax - vmin)
        else:
            alphas = np.ones_like(snr)

        # Create RGBA colors from colormap with per-point alpha
        colors = cmap(norm(snr))
        colors[:, 3] = alphas
        sc_pol = ax_pol.scatter(np.radians(pol_angles), r_from_centroid, c=colors,
                edgecolors='k', linewidths=0.1, s=40)
        ax_pol.set_thetamin(0)
        ax_pol.set_thetamax(180)
        ax_pol.set_yticklabels([])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_pol, label='SNR', pad=0.1)
        ax_pol.set_title('Radial Distance from FTS Centroid vs Pol Angle\n(folded, color=SNR)', va='bottom')
        plt.tight_layout()
        plt.show()

    else:
        print("Centroid fit failed, skipping focal plane and polar plots.")
