# This code is from VAPE, Packer LAb 06/03/2022

import numpy as np
import pandas as pd
import json
# import tifffile as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import bisect
import copy
from scipy import stats
import scipy.io as spio
from scipy import signal
import pickle

# global plotting params
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
sns.set()
sns.set_style('white')


def intersect(lst1, lst2):
    return list(set(lst1) & set(lst2)) 

def dfof(arr):
    '''takes 1d list or array or 2d array and returns dfof array of same
       dim (JR 2019) This is extraordinarily slow, use dfof2'''

    if type(arr) is list or type(arr) == np.ndarray and len(arr.shape) == 1:
        F = np.mean(arr)
        dfof_arr = [((f - F) / F) * 100 for f in arr]

    elif type(arr) == np.ndarray and len(arr.shape) == 2:
        dfof_arr = []
        for trace in arr:
            F = np.mean(trace)
            dfof_arr.append([((f - F) / F) * 100 for f in trace])

    else:
        raise NotImplementedError('input type not recognised')

    return np.array(dfof_arr)


def dfof2(flu):
    '''
    delta f over f, this function is orders of magnitude faster 
    than the dumb one above takes input matrix flu 
    (num_cells x num_frames)
    (JR 2019)

    '''

    flu_mean = np.mean(flu, 1)
    flu_mean = np.reshape(flu_mean, (len(flu_mean), 1))
    return (flu - flu_mean) / flu_mean


def get_tiffs(path):

    tiff_files = []
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.tiff'):
            tiff_files.append(os.path.join(path, file))

    return tiff_files


def s2p_loader(s2p_path, subtract_neuropil=True, neuropil_coeff=0.7):

    found_stat = False

    for root, dirs, files in os.walk(s2p_path):

        for file in files:

            if file == 'F.npy':
                all_cells = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'Fneu.npy':
                neuropil = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'iscell.npy':
                is_cells = np.load(os.path.join(root, file), 
                                   allow_pickle=True)[:, 0]
                is_cells = np.ndarray.astype(is_cells, 'bool')
                print('loading {} traces labelled as cells'
                      .format(sum(is_cells)))
            elif file == 'spks.npy':
                spks = np.load(os.path.join(root, file), allow_pickle=True)
            elif file == 'stat.npy':
                stat = np.load(os.path.join(root, file), allow_pickle=True)
                found_stat = True

    if not found_stat:
        raise FileNotFoundError('Could not find stat, '
                                'this is likely not a suit2p folder')
    for i, s in enumerate(stat):
        s['original_index'] = i

    all_cells = all_cells[is_cells, :]
    neuropil = neuropil[is_cells, :]
    spks = spks[is_cells, :]
    stat = stat[is_cells]


    if not subtract_neuropil:
        return all_cells, spks, stat

    else:
        print('subtracting neuropil with a coefficient of {}'
              .format(neuropil_coeff))
        neuropil_corrected = all_cells - neuropil * neuropil_coeff
        return neuropil_corrected, spks, stat


def correct_s2p_combined(s2p_path, n_planes):

    len_count = 0
    for i in range(n_planes):

        iscell = np.load(os.path.join(s2p_path, 'plane{}'.format(i), 
                                     'iscell.npy'), allow_pickle=True)

        if i == 0:
            allcells = iscell
        else:
            allcells = np.vstack((allcells, iscell))

        len_count += len(iscell)

    combined_iscell = os.path.join(s2p_path, 'combined', 'iscell.npy')

    ic = np.load(combined_iscell, allow_pickle=True)
    assert ic.shape == allcells.shape
    assert len_count == len(ic)

    np.save(combined_iscell, allcells)


def read_fiji(csv_path):
    '''reads the csv file saved through plot z axis profile in fiji'''

    data = []

    with open(csv_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            data.append(float(row[0].split(',')[1]))

    return np.array(data)


# def save_fiji(arr):  # commented out by Thijs for compatibility
#     '''saves numpy array in current folder as fiji friendly tiff'''
#     tf.imsave('Vape_array.tiff', arr.astype('int16'))

def clean_traces(signalMain):
    '''takes a 2d array of traces and returns a 2d array of cleaned traces
       (HA 2023)'''

    signalMain[np.logical_not(pd.isna(signalMain))] = signal.detrend(signalMain[np.logical_not(pd.isna(signalMain))])

    return np.array(signalMain)

def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]

def pade_approx_norminv(p):
    q = math.sqrt(2*math.pi) * (p - 1/2) - (157/231) * math.sqrt(2) * \
        math.pi**(3/2) * (p - 1/2)**3
    r = 1 - (78/77) * math.pi * (p - 1/2)**2 + (241 * math.pi**2 / 2310) * \
        (p - 1/2)**4
    return q/r

def d_prime(hit_rate, false_alarm_rate):
    return pade_approx_norminv(hit_rate) - \
        pade_approx_norminv(false_alarm_rate)

# Functions for reading in data from .paq files
def paq_data(paq, chan_name, threshold=1, threshold_ttl=False, plot=False):
    '''
    Do not include any exclusion of data in this function
    returns the data in paq (from paq_read) from channel: chan_names
    if threshold_tll: returns sample that trigger occured on
    '''

    chan_idx = paq['chan_names'].index(chan_name)
    data = paq['data'][chan_idx, :]
    if threshold_ttl:
        data = threshold_detect(data,threshold)

    if plot:
        if threshold_ttl:
            plt.plot(data, np.ones(len(data)), '.')
        else:
            plt.plot(data)
    return data

def shutter_start_frame(paq=None, stim_chan_name=None, frame_clock=None,
                     stim_times=None, plane=0, n_planes=1,threshold=1,):
    ''' Only differences is NOT NEXT FRAME'''
    '''Returns the frames from a frame_clock that a stim occured on.
       Either give paq and stim_chan_name as arugments if using 
       unprocessed paq. 
       Or predigitised frame_clock and stim_times in reference frame
       of that clock

    '''

    if frame_clock is None:
        frame_clock = paq_data(paq, 'prairieFrame',threshold, threshold_ttl=True)
        stim_times = paq_data(paq, stim_chan_name,threshold, threshold_ttl=True)

    stim_times = [stim for stim in stim_times if stim < np.nanmax(frame_clock)]
    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
#         frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock[plane::n_planes])
#                      if sample - stim > 0)
        frame = next(i-1 for i, sample in enumerate(frame_clock[plane::n_planes])
                     if sample - stim > 0)
        frames.append(frame)
    frames = np.transpose(frames)
    return frames

def stim_start_frame_Dual2Psetup(frame_clock, stim_times):
    # used in the analysis code.
    #
    ''' Returns the frames from the frame_cloc immediately preeceding stim.
    This code needs the frame_clock from txt_file
    '''
    if isinstance(frame_clock, np.ndarray)==False:
        frame_clock = frame_clock.values
    if isinstance(stim_times, np.ndarray)==False:
        stim_times = stim_times.values

    plane=0
    n_planes=1 # might be useful in the future
    fs = 20000
    stim_times = np.round(stim_times*fs)
    stim_times = [stim if (stim < np.nanmax(frame_clock)) or np.isnan(stim) else stim_times[i] for i, stim in enumerate(stim_times)]

    frames = []

    for stim in stim_times:
        if ~np.isnan(stim) and (stim < np.nanmax(frame_clock)) and (stim > np.nanmin(frame_clock)):
        # the sample time of the frame immediately preceeding stim
#         frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock[plane::n_planes])
#                      if sample - stim > 0)
            frame = next(i-1 for i, sample in enumerate(frame_clock[plane::n_planes])
                        if sample - stim > 0)
        else:
            frame = np.nan

        frames.append(frame)
    return (frames)

def stim_start_frame(paq=None, stim_chan_name=None, frame_clock=None,
                     stim_times=None, threshold=1, plane=0, n_planes=1):
    # used in _analysis code.
    '''Returns the frames from a frame_clock that a stim occured on.
       Either give paq and stim_chan_name as arugments if using 
       unprocessed paq. 
       Or predigitised frame_clock and stim_times in reference frame
       of that clock

    '''

    if frame_clock is None:
        frame_clock = paq_data(paq, 'frame_clock',threshold, threshold_ttl=True)
        stim_times  = paq_data(paq, stim_chan_name,threshold, threshold_ttl=True)
        interStimFrameMin = 30 # 30 frames = 1 second
    elif frame_clock == 'BehOnly':
        frame_clock = paq_data(paq, 'pupilLoopback',threshold, threshold_ttl=True)
        stim_times  = paq_data(paq, stim_chan_name,threshold, threshold_ttl=True)
        interStimFrameMin = 0 # 30 frames = 1 second

    stim_times = [stim for stim in stim_times if stim < np.nanmax(frame_clock)]

    frames = []

    for stim in stim_times:
        # the sample time of the frame immediately preceeding stim
#         frame = next(frame_clock[i-1] for i, sample in enumerate(frame_clock[plane::n_planes])
#                      if sample - stim > 0)
        frame = next(i-1 for i, sample in enumerate(frame_clock[plane::n_planes])
                     if sample - stim > 0)
        frames.append(frame)

     # Exclude frames that are too close together
    first_ind = np.where(np.diff(frames)>interStimFrameMin)
    first_ind = np.concatenate(([0], first_ind[0]+1))
    frames = np.array(frames)
    frames = frames[first_ind]

    return (frames)

def myround(x, base=5):
    '''allow rounding to nearest base number for
       use with multiplane stack slicing'''

    return base * round(x/base)

def tseries_finder(tseries_lens, frame_clock, paq_rate=20000):

    ''' Finds chunks of frame clock that correspond to the tseries in 
        tseries lens
        tseries_lens -- list of the number of frames each tseries you want 
                        to find contains
        frame_clock  -- thresholded times each frame recorded in paqio occured
        paq_rate     -- input sampling rate of paqio

        '''

    # frame clock recorded in paqio, includes TTLs from cliking 'live' 
    # and foxy extras
    clock = frame_clock / paq_rate

    # break the recorded frame clock up into individual aquisitions
    # where TTLs are seperated by more than 1s
    gap_idx = np.where(np.diff(clock) > 1)
    gap_idx = np.insert(gap_idx, 0, 0)
    gap_idx = np.append(gap_idx, len(clock))
    chunked_paqio = np.diff(gap_idx)

    # are each of the frames recorded by the frame clock actually 
    # in processed tseries?
    real_frames = np.zeros(len(clock))
    # the max number of extra frames foxy could spit out
    foxy_limit = 20
    # the number of tseries blocks that have already been found
    series_found = 0
    # count how many frames have been labelled as real or not
    counter = 0

    for chunk in chunked_paqio:
        is_tseries = False

        # iterate through the actual length of each analysed tseries
        for idx, ts in enumerate(tseries_lens): 
            # ignore previously found tseries
            if idx < series_found:
                continue

            # the frame clock block matches the number of frames in a tseries
            if chunk >= ts and chunk <= ts + foxy_limit:
                # this chunk of paqio clock is a recorded tseries
                is_tseries = True
                # advance the number of tseries found so they are not 
                # detected twice
                series_found += 1
                break

        if is_tseries:
            # foxy bonus frames
            extra_frames = chunk - ts
            # mark tseries frames as real
            real_frames[counter:counter+ts] = 1
            # move the counter on by the length of the real tseries
            counter += ts
            # set foxy bonus frames to not real
            real_frames[counter:counter+extra_frames] = 0
            # move the counter along by the number of foxy bonus frames
            counter += extra_frames

        else:
            # not a tseries so just move the counter along by the chunk 
            # of paqio clock
            # this could be wrong, not sure if i've fixed the ob1 error,
            # go careful
            counter += chunk + 1

    real_idx = np.where(real_frames == 1)

    return frame_clock[real_idx]

def trace_splitter(trace,t_starts, pre_frames, post_frames):
    '''Split a fluoresence matrix into trial by trial array

       flu -- fluoresence matrix [num_cells x num_frames]
       t_starts -- the time each frame started
       pre_frames -- the number of frames before t_start
                     to include in the trial
       post_frames --  the number of frames after t_start
                       to include in the trial

       returns 
       trial_flu -- trial by trial array 
                    [num_cells x trial frames x num_trials]

       '''
    initial=True
    trial_trace = np.array
    for trial, t_start in enumerate(t_starts):
        # the trial occured before imaging started
    
        if (t_start-pre_frames) > 0: #ignore first trial 
        #     print('prb')
        #     continue
            flu_chunk = trace[t_start-pre_frames:t_start+post_frames]

            if initial==True:
                trial_trace = flu_chunk
                initial = False
            else:
                trial_trace = np.dstack((trial_trace, flu_chunk))

    return trial_trace

def flu_splitter(flu,t_starts, pre_frames, post_frames):
    '''Split a fluoresence matrix into trial by trial array

       flu -- fluoresence matrix [num_cells x num_frames]
       t_starts -- the time each frame started
       pre_frames -- the number of frames before t_start
                     to include in the trial
       post_frames --  the number of frames after t_start
                       to include in the trial

       returns 
       trial_flu -- trial by trial array 
                    [num_cells x trial frames x num_trials]

       '''
    initial = True
    trial_flu = np.array
    for trial, t_start in enumerate(t_starts):
        # the trial occured before imaging started
    
        if ((t_starts[trial]-pre_frames) > 0 ) & ((t_start+post_frames)<flu.shape[1]):
            #     print('prb')
            #     continue
            flu_chunk = flu[:, t_start-pre_frames:t_start+post_frames]

            if initial == True:
                trial_flu = flu_chunk
                initial = False
            else:
                trial_flu = np.dstack((trial_flu, flu_chunk))

    return trial_flu

def flu_splitter2(flu, stim_times, frames_ms, pre_frames=10, post_frames=30):

    stim_idxs = stim_start_frame_mat(stim_times, frames_ms, debug_print=False)

    stim_idxs = stim_idxs[:, np.where((stim_idxs[0, :]-pre_frames > 0) &
                                      (stim_idxs[0, :] + post_frames 
                                      < flu.shape[1]))[0]]

    n_trials = stim_idxs.shape[1]
    n_cells = frames_ms.shape[0]

    for i, shift in enumerate(np.arange(-pre_frames, post_frames)):
        if i == 0:
            trial_idx = stim_idxs + shift
        else:
            trial_idx = np.dstack((trial_idx, stim_idxs + shift))

    tot_frames = pre_frames + post_frames
    trial_idx = trial_idx.reshape((n_cells, n_trials*tot_frames))

    flu_trials = []
    for i, idxs in enumerate(trial_idx):
        idxs = idxs[~np.isnan(idxs)].astype('int')
        flu_trials.append(flu[i, idxs])

    n_trials_valid = len(idxs)
    flu_trials = np.array(flu_trials).reshape(
        (n_cells, int(n_trials_valid/tot_frames), tot_frames))

    return flu_trials

def flu_splitter3(flu, stim_times, frames_ms, pre_frames=10, post_frames=30):

    stim_idxs = stim_start_frame_mat(stim_times, frames_ms, debug_print=False)

    # not 100% sure about this line, keep an eye
    stim_idxs[:, np.where((stim_idxs[0, :]-pre_frames <= 0) |
                          (stim_idxs[0, :] + post_frames 
                           >= flu.shape[1]))[0]] = np.nan

    n_trials = stim_idxs.shape[1]
    n_cells = frames_ms.shape[0]

    for i, shift in enumerate(np.arange(-pre_frames, post_frames)):
        if i == 0:
            trial_idx = stim_idxs + shift
        else:
            trial_idx = np.dstack((trial_idx, stim_idxs + shift))

    tot_frames = pre_frames + post_frames
    trial_idx = trial_idx.reshape((n_cells, n_trials*tot_frames))

    # flu_trials = np.repeat(np.nan, n_cells*n_trials*tot_frames)
    # flu_trials = np.reshape(flu_trials, (n_cells, n_trials, tot_frames))
    flu_trials = np.full_like(trial_idx, np.nan)
    # iterate through each cell and add trial frames
    for i, idxs in enumerate(trial_idx):

        non_nan = ~np.isnan(idxs)
        idxs = idxs[~np.isnan(idxs)].astype('int')
        flu_trials[i, non_nan] = flu[i, idxs]

    flu_trials = np.reshape(flu_trials, (n_cells, n_trials, tot_frames))
    return flu_trials

def closest_frame_before(clock, t):
    ''' Returns the idx of the frame immediately preceeding 
        the time t.
        Frame clock must be digitised and expressed
        in the same reference frame as t.
        '''
    subbed = np.array(clock) - t
    return np.where(subbed < 0, subbed, -np.inf).argmax()

def closest_frame(clock, t):
    ''' Returns the idx of the frame closest to  
        the time t. 
        Frame clock must be digitised and expressed
        in the same reference frame as t.
        '''
    subbed = np.array(clock) - t
    return np.argmin(abs(subbed))

def test_responsive(flu, frame_clock, stim_times, pre_frames=10, 
                    post_frames=10, offset=0, testType = 'ttest'):
    ''' Tests if cells in a fluoresence array are significantly responsive 
        to a stimulus

        Inputs:
        flu -- fluoresence matrix [n_cells x n_frames] likely dfof from suite2p
        frame_clock -- timing of the frames, must be digitised and in same 
                       reference frame as stim_times
        stim_times -- times that stims to test responsiveness on occured, 
                      must be digitised and in same reference frame 
                      as frame_clock
        pre_frames -- the number of frames before the stimulus occured to 
                      baseline with
        post_frames -- the number of frames after stimulus to test differnece 
                       compared
                       to baseline
        offset -- the number of frames to offset post_frames from the 
                  stimulus, so don't take into account e.g. stimulus artifact

        Returns:
        pre -- matrix of fluorescence values in the pre_frames period 
               [n_cells x n_frames]
        post -- matrix of fluorescence values in the post_frames period 
                [n_cells x n_frames]
        pvals -- vector of pvalues from the significance test [n_cells]

        '''

    n_frames = flu.shape[1]

    pre_idx = np.repeat(False, n_frames)
    post_idx = np.repeat(False, n_frames)

    # keep track of the previous stim frame to warn against overlap
    prev_frame = 0

    for i, stim_frame in enumerate(stim_times):

        if stim_frame-pre_frames <= 0 or stim_frame+post_frames+offset \
           >= n_frames:
            continue
        elif stim_frame - pre_frames <= prev_frame:
            print('WARNING: STA for stim number {} overlaps with the '
                  'previous stim pre and post arrays can not be '
                  'reshaped to trial by trial'.format(i))

        prev_frame = stim_frame

        pre_idx[stim_frame-pre_frames: stim_frame] = True
        post_idx[stim_frame+offset: stim_frame+post_frames+offset] = True

    pre = flu[:, pre_idx]
    post = flu[:, post_idx]

    if testType =='wilcoxon':
        _, pvals = stats.ttest_ind(pre, post, axis=1)
    elif testType =='ttest':
        _, pvals = stats.wilcoxon(pre, post, axis=1)

    return pre, post, pvals

def build_flu_array(run, stim_times, pre_frames=10, post_frames=50,
                    use_spks=False, use_comps=False, is_prereward=False):

    ''' converts [n_cells x n_frames] matrix to trial by trial array
        [n_cells x n_trials x pre_frames+post_frames]

        Inputs:
        run -- BlimpImport object with attributes flu and frames_ms
        stim_times -- times of trial start stims, should be same
                      reference frame as frames_ms
        pre_frames -- number of frames before stim to include in
                      trial
        post_frames -- number of frames after stim to include 
                       in trial

        Returns:
        flu_array -- array [n_cells x n_trials x pre_frames+post_frames]

    '''

    if use_spks:
        flu = run.spks
    elif use_comps:
        flu = run.comps
    else:
        flu = run.flu

    if is_prereward:
        frames_ms = run.frames_ms_pre
    else:
        frames_ms = run.frames_ms

    # split flu matrix into trials based on stim time
    flu_array = flu_splitter3(flu, stim_times, frames_ms,
                              pre_frames=pre_frames, post_frames=post_frames)

    return flu_array

def averager(array_list, pre_frames=10, post_frames=50, offset=0, 
             trial_filter=None, plot=False, fs=5):

    ''' Averages list of trial by trial fluoresence arrays and can 
        visualise results

        Inputs:
        array_list -- list of tbt fluoresence arrays
        pre_frames -- number of frames before stim to include in
                      trial
        post_frames -- number of frames after stim to include 
                       in trial
        offset -- number of frames to offset post_frames to avoid artifacts
        trial_filter -- list of trial indexs to include 
        plot -- whether to plot result
        fs -- frame rate / plane

        Returns:
        session_average -- mean array [n_sessions x pre_frames+post_frames]
        scaled_average -- same as session average but all traces start 
                          at dfof = 0
        grand_average -- average across all sessions [pre_frames + post_frames]
        cell_average -- list with length n_sessions contains arrays 
                        [n_cells x pre_frames+post_frames]

        '''

    if trial_filter:
        assert len(trial_filter) == len(array_list)
        array_list = [arr[:, filt, :]
                      for arr, filt in zip(array_list, trial_filter)]

    n_sessions = len(array_list)

    cell_average = [np.nanmean(k, 1) for k in array_list]

    session_average = np.array([np.nanmean(np.nanmean(k, 0), 0)
                               for k in array_list])

    scaled_average = np.array([session_average[i, :] - session_average[i, 0]
                               for i in range(n_sessions)])

    grand_average = np.nanmean(scaled_average, 0)

    if plot:
        x_axis = range(len(grand_average))
        plt.plot(x_axis, grand_average)
        plt.plot(x_axis[0:pre_frames],
                 grand_average[0:pre_frames], color='red')
        plt.plot(x_axis[pre_frames+offset:pre_frames+offset
                 +(post_frames-offset)], grand_average[pre_frames+offset:
                 pre_frames+offset+(post_frames-offset)], color='red')
        for s in scaled_average:
            plt.plot(x_axis, s, alpha=0.2, color='grey')

        plt.ylabel(r'$\Delta $F/F')
        plt.xlabel('Time (Seconds)')
        plt.axvline(x=pre_frames-1, ls='--', color='red')

    return session_average, scaled_average, grand_average, cell_average

def lick_binner(pathname,trial_start, stChanName, stimulation=True ):
    ''' makes new easytest binned lick variable in run object '''

    if stimulation:
        paqData = pd.read_pickle (pathname +'paq-data.pkl')
    else:
        paqData = pd.read_pickle (pathname +'training-paq-data.pkl')
    licks = paq_data (paqData, stChanName, 1, threshold_ttl=True)

    binned_licks = []

    for i, t_start in enumerate(trial_start):
        if i == len(trial_start) - 1:
            t_end = np.inf
        else:
            t_end = trial_start[i+1]

        trial_idx = np.where((licks >= t_start) & (licks <= t_end))[0]

        trial_licks = licks[trial_idx] - t_start

        binned_licks.append(trial_licks)

    licks = licks
    # attribute already exists called 'binned_licks' and cannot overwrite it
    binned_licks_easytest = binned_licks

    return licks, binned_licks

def prepost_diff(array_list, pre_frames=10,
                 post_frames=50, offset=0, filter_list=None):

    n_sessions = len(array_list)

    if filter_list:
        array_list = [array_list[i][:, filter_list[i], :]
                      for i in range(n_sessions)]

    session_average, _, _, cell_average = averager(
        array_list, pre_frames, post_frames)

    post = np.nanmean(
                      session_average[:, pre_frames+offset:pre_frames+offset
                      +(post_frames-offset)], 1
                     )
    pre = np.nanmean(session_average[:, 0:pre_frames], 1)

    return post - pre

def raster_plot(arr, y_pos=1, color=np.random.rand(3,), alpha=1,
                marker='.', markersize=12, label=None):

    plt.plot(arr, np.ones(len(arr)) * y_pos, marker,
             color=color, alpha=alpha, markersize=markersize,
             label=label)

def get_spiral_start(x_galvo, debounce_time):
    
    """ Get the sample at which the first spiral in a trial began 
    
    Experimental function involving lots of magic numbers
    to detect spiral onsets.
    Failures should be caught by assertion at end
    Inputs:
    x_galvo -- x_galvo signal recorded in paqio
    debouce_time -- length of time (samples) encapulsating a whole trial
                    ensures only spiral at start of trial is captured
    
    """
    #x_galvo = np.round(x_galvo, 2)
    x_galvo = my_floor(x_galvo, 2)
    
    # Threshold above which to determine signal as onset of square pulse
    square_thresh = 0.02
    # Threshold above which to consider signal a spiral (empirically determined)
    diff_thresh = 10
    
    # remove noise from parked galvo signal
    x_galvo[x_galvo < -0.5] = -0.6
    
    diffed = np.diff(x_galvo)
    # remove the onset of galvo movement from f' signal
    diffed[diffed > square_thresh] = 0
    diffed = non_zero_smoother(diffed, window_size=200)
    diffed[diffed>30] = 0
    
    # detect onset of sprials
    spiral_start = threshold_detect(diffed, diff_thresh)
    
    if len(spiral_start) == 0:
        print('No spirals found')
        return None
    else:
        # Debounce to remove spirals that are not the onset of the trial
        spiral_start = spiral_start[np.hstack((np.inf, np.diff(spiral_start))) > debounce_time]
        n_squares = len(threshold_detect(x_galvo, -0.5))
        assert len(spiral_start) == n_squares, \
        'spiral_start has len {} but there are {} square pulses'.format(len(spiral_start), n_squares)
        return spiral_start

def non_zero_smoother(arr, window_size=200):
    
    """ Smooths an array by changing values to the number of
        non-0 elements with window
        
        """
    
    windows = np.arange(0, len(arr), window_size)
    windows = np.append(windows, len(arr))

    for idx in range(len(windows)):

        chunk_start = windows[idx]
        
        if idx == len(windows) - 1:
            chunk_end = len(arr)
        else:
            chunk_end = windows[idx+1]
            
        arr[chunk_start:chunk_end] = np.count_nonzero(arr[chunk_start:chunk_end])
    
    return arr

def my_floor(a, precision=0):
    # Floors to a specified number of dps
    return np.round(a - 0.5 * 10**(-precision), precision)

def get_trial_frames(clock, start, pre_frames, post_frames, paq_rate=2000, fs=30):

    # The frames immediately preceeding stim
    start_idx = closest_frame_before(clock, start)
    frames = np.arange(start_idx-pre_frames, start_idx+post_frames)
    
    # Is the trial outside of the frame clock
    is_beyond_clock = np.max(frames) >= len(clock) or np.min(frames) < 0
    
    if is_beyond_clock:
        return None, None
    
    frame_to_start = (start - clock[start_idx]) / paq_rate  # time (s) from frame to trial_start
    frame_time_diff = np.diff(clock[frames]) / paq_rate  # ifi (s)
    
    # did the function find the correct frame
    is_not_correct_frame = clock[start_idx+1]  < start or clock[start_idx] > start
    # the nearest frame to trial start was not during trial
    # if the time to the nearest frame is less than upper bound of inter-frame-interval
    trial_not_running = frame_to_start > 1/(fs-1)
    frames_not_consecutive = np.max(frame_time_diff) > 1/(fs-1)
    
    if trial_not_running or frames_not_consecutive:
        return None, None
    
    return frames, start_idx

def adamiser(string):
    words = string.split(' ')
    n_pleases = int(len(words) / 10)
    
    for please in range(n_pleases):
        idx = randrange(len(words))
        words.insert(idx, 'please')
        
    n_caps = int(len(words) / 3)
    for cap in range(n_caps):
        idx = randrange(len(words))
        words[idx] = words[idx].upper()
        
    return ' '.join(words)

def between_two_hits(idxs, easy_idxs, easy_outcome):
    
    assert len(easy_idxs) == len(easy_outcome)
    
    # Next easy trial from each test trial
    closest_after = np.array([bisect.bisect_left(easy_idxs, idx) for idx in idxs])
    # Previous easy trial from each test trial
    closest_before = closest_after - 1
    # Test trials before the first easy trial should have both previous and next
    # as the first easy trial
    closest_before[closest_before==-1] = 0
    # Test trials after the last easy trial should have both previous and next
    # as the last easy trial
    closest_after[idxs>easy_idxs[-1]] = len(easy_idxs)-1
    
    assert len(idxs) == len(closest_before) == len(closest_after)
    
    between_two = []
    for before, after in zip(closest_before, closest_after):
        if easy_outcome[before] and easy_outcome[after] == 'hit':
            between_two.append(True)
        else:
            between_two.append(False)
    
    assert len(between_two) == len(idxs)
    
    return between_two

def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y

def build_frames_ms(flu, cell_plane, paqio_frames, aligner, num_planes):

    ''' builds frames_ms matrix (see preprocess_flu)
        aligner -- rsync object from rsync_aligner
   
        '''
    # convert paqio recorded frames to pycontrol ms
    ms_vector = aligner.B_to_A(paqio_frames) # flat list of plane 
                                             # times
    if num_planes == 1:
        return ms_vector
    
    # matrix of frame times in ms for each fluorescent value 
    # in the flu matrix
    frames_ms = np.empty(flu.shape)
    frames_ms.fill(np.nan)
 
    # mark each frame with a time in ms based on its plane
    for plane in range(num_planes):
        frame_times = ms_vector[plane::num_planes]
        plane_idx = np.where(cell_plane==plane)[0]
        frames_ms[plane_idx, 0:len(frame_times)] = frame_times

    return frames_ms


class LoadMat():
    def __init__(self, filename):

        '''
        This function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        
        Mostly stolen from some hero https://stackoverflow.com/a/8832212
        
        '''

        self.dict_ = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

        self._check_keys()

    def _check_keys(self):

        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in self.dict_:
            if isinstance(self.dict_[key], spio.matlab.mio5_params.mat_struct):
                self.dict_[key] = self._todict(self.dict_[key])

    @staticmethod
    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict_ = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict_[strg] = LoadMat._todict(elem)
            else:
                dict_[strg] = elem
        return dict_

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import scipy.io as sio
import platform

class local_analysis:
    def __init__(self, data_path):
        """
        Inicializa el análisis con datos locales.
        
        Args:
            data_path (str): Ruta al directorio que contiene los archivos PAQ y TIFF
        """
        # Configurar rutas según el sistema
        if platform.node() == 'WIN-AL012':
            print("Computer: Candela Windows")
            self.suite2pOutputPath = data_path
            self.recordingListPath = data_path
            self.rawPath = data_path
            self.rootPath = data_path
        else:
            print('Computer setting is not set.')
            self.suite2pOutputPath = 'N/A'
            self.recordingListPath = data_path
            self.rawPath = data_path
            self.rootPath = data_path

        self.analysisPath = os.path.join(self.rootPath, 'analysis')
        self.figsPath = os.path.join(self.rootPath, 'figs')
        self.data_path = data_path
        self.recordingList = self._create_recording_list()
        
    def _create_recording_list(self):
        """
        Crea una lista de grabaciones basada en los archivos PAQ, TIFF, Block.mat y Timeline.mat encontrados.
        """
        # Encontrar todos los archivos PAQ en el directorio TwoP
        twoP_path = os.path.join(self.data_path, 'TwoP')
        paq_files = glob.glob(os.path.join(twoP_path, "*.paq"))
        
        # Crear DataFrame para almacenar la información
        records = []
        
        for paq_file in paq_files:
            # Extraer información del nombre del archivo
            base_name = os.path.basename(paq_file)
            parts = base_name.split('_')
            
            if len(parts) >= 3:
                recording_date = parts[0]
                animal_id = parts[1]
                recording_id = parts[2].split('_')[0]  # Eliminar '_paq' y extensión
                
                # Crear el nombre de la sesión
                session_name = f"{recording_date}_{recording_id}_{animal_id}"
                
                # Buscar archivo TIFF correspondiente
                tiff_pattern = f"{recording_date}_t-*_Cycle*_Ch2.tif"
                tiff_files = glob.glob(os.path.join(twoP_path, "**", tiff_pattern), recursive=True)
                
                # Buscar archivo Block.mat
                block_pattern = f"{recording_date}_{recording_id}_{animal_id}_Block.mat"
                block_files = glob.glob(os.path.join(self.data_path, block_pattern))
                
                # Buscar archivo Timeline.mat
                timeline_pattern = f"{recording_date}_{recording_id}_{animal_id}_Timeline.mat"
                timeline_files = glob.glob(os.path.join(self.data_path, timeline_pattern))
                
                # Cargar datos del Block.mat si existe
                block_data = None
                if block_files:
                    try:
                        block_data = sio.loadmat(block_files[0])
                    except Exception as e:
                        print(f"Error al cargar Block.mat: {e}")
                
                # Cargar datos del Timeline.mat si existe
                timeline_data = None
                if timeline_files:
                    try:
                        timeline_data = sio.loadmat(timeline_files[0])
                    except Exception as e:
                        print(f"Error al cargar Timeline.mat: {e}")
                
                # Determinar si es una sesión de aprendizaje
                date = datetime.strptime(recording_date, '%Y-%m-%d')
                learning = False  # Por defecto, no es una sesión de aprendizaje
                
                record = {
                    'recordingDate': recording_date,
                    'animalID': animal_id,
                    'recordingID': recording_id,
                    'sessionName': session_name,
                    'learningData': learning,
                    'path': self.data_path,
                    'twoP': len(tiff_files) > 0,  # True si existe archivo TIFF
                    'paqFileName': paq_file,
                    'imagingTiffFileNames': tiff_files[0] if tiff_files else None,
                    'blockData': block_data,
                    'timelineData': timeline_data,
                    'hasBlock': len(block_files) > 0,
                    'hasTimeline': len(timeline_files) > 0,
                    'sessionNameWithPath': os.path.join(self.data_path, f"{session_name}_Block.mat"),
                    'eventTimesExtracted': 0,  # Inicializar como 0
                    'eventTimesPath': '',  # Inicializar como string vacío
                    'variance': np.nan  # Inicializar como NaN
                }
                records.append(record)
        
        # Crear DataFrame
        df = pd.DataFrame(records)
        
        # Ordenar por fecha y animal
        if not df.empty:
            df = df.sort_values(['recordingDate', 'animalID'])
            
            # Añadir rutas de análisis
            df['analysispathname'] = np.nan
            df['filepathname'] = np.nan
            
            for ind, recordingDate in enumerate(df.recordingDate):
                # Crear la ruta del archivo
                filepathname = os.path.join(df.path[ind], df.recordingID[ind])
                df.loc[ind, 'filepathname'] = filepathname
                
                # Crear la ruta de análisis
                analysispathname = os.path.join(
                    self.analysisPath,
                    f"{df.recordingDate[ind]}_{df.animalID[ind]}_{df.recordingID[ind]}"
                )
                df.loc[ind, 'analysispathname'] = analysispathname + '\\'
                
                # Crear el directorio de análisis si no existe
                if not os.path.exists(analysispathname):
                    os.makedirs(analysispathname)
                
                # Inicializar la ruta del archivo CSV de eventos
                df.loc[ind, 'eventTimesPath'] = os.path.join(
                    analysispathname,
                    f"{df.sessionName[ind]}_CorrectedeventTimes.csv"
                )
        
        return df

def get_info(data_path="C:\\Users\\Lak Lab\\Documents\\paqtif\\2025-05-20"):
    """
    Función auxiliar para obtener el objeto info.
    
    Args:
        data_path (str): Ruta al directorio que contiene los archivos PAQ y TIFF
        
    Returns:
        local_analysis: Objeto con la información de las grabaciones
    """
    return local_analysis(data_path) 

import tifffile
import numpy as np
import os
from paq2py import paq_read

def get_tiff_frames(tiff_path):
    """
    Efficiently gets the number of frames from a TIFF file.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        # Try to get the number of frames from ImageDescription first
        try:
            desc = tif.pages[0].tags['ImageDescription'].value
            import re
            # Look for the pattern [n,] in the description
            n_frames = re.search(r'\[(\d+),', desc)
            if n_frames:
                return int(n_frames.group(1))
        except:
            pass
        
        # If we can't get it from ImageDescription, count the pages
        return len(tif.pages)

def show_paq_info(paq_path):
    """
    Shows the information contained in a PAQ file.
    
    Args:
        paq_path (str): Path to the PAQ file
    """
    print(f"Reading PAQ file: {paq_path}")
    paq_data = paq_read(paq_path, plot=False)
    
    print("\nPAQ file information:")
    print(f"Sampling rate: {paq_data['rate']} Hz")
    print(f"Number of frames: {paq_data['data'].shape[1]}")
    print(f"Number of channels: {len(paq_data['chan_names'])}")
    
    print("\nAvailable channels:")
    for i, (chan_name, unit) in enumerate(zip(paq_data['chan_names'], paq_data['units'])):
        print(f"{i+1}. {chan_name} ({unit})")
    
    print("\nHardware lines:")
    for hw_chan in paq_data['hw_chans']:
        print(f"- {hw_chan}")
    
    # Show basic statistics for each channel
    print("\nChannel statistics:")
    for i, chan_name in enumerate(paq_data['chan_names']):
        data = paq_data['data'][i]
        print(f"\n{chan_name}:")
        print(f"  Minimum: {np.min(data):.2f}")
        print(f"  Maximum: {np.max(data):.2f}")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Standard deviation: {np.std(data):.2f}")

def adjust_paq_to_tiff(tiff_path, paq_path, output_paq_path=None):
    """
    Adjusts the PAQ file to match the number of frames in the TIFF file,
    based on the number of frames in the 2P channel.
    
    Args:
        tiff_path (str): Path to the TIFF file
        paq_path (str): Path to the original PAQ file
        output_paq_path (str, optional): Path where to save the new PAQ file.
                                       If not specified, uses the original name with '_adjusted' added.
    
    Returns:
        str: Path to the newly created PAQ file
    """
    print("Getting number of frames from TIFF...")
    tiff_frames = get_tiff_frames(tiff_path)
    print(f"Number of frames in TIFF file: {tiff_frames}")
    
    # Check if the number of frames seems reasonable
    if tiff_frames < 1000:  # A 20GB file should have many more frames
        print("WARNING: The detected number of frames seems very low for a 20GB file.")
        print("Trying alternative method...")
        with tifffile.TiffFile(tiff_path) as tif:
            # Print diagnostic information
            print("\nTIFF file information:")
            print(f"File size: {os.path.getsize(tiff_path) / (1024**3):.2f} GB")
            print(f"Number of pages: {len(tif.pages)}")
            print("First page metadata:")
            for tag in tif.pages[0].tags.values():
                print(f"  {tag.name}: {tag.value}")
            
            # Use number of pages as number of frames
            tiff_frames = len(tif.pages)
            print(f"\nUsing number of pages as number of frames: {tiff_frames}")
    
    print("Reading PAQ file...")
    paq_data = paq_read(paq_path, plot=False)
    
    # Get number of frames in PAQ
    paq_frames = paq_data["data"].shape[1]
    print(f"Number of frames in original PAQ file: {paq_frames}")
    
    # Find the index of the 2P channel
    try:
        idx_2p = paq_data["chan_names"].index("2p_frame")
    except ValueError:
        print("Error: Channel '2p_frame' not found in PAQ file")
        return paq_path
    
    # Count 2P channel frames
    threshold_volts = 2.5
    frame_count_2p = np.flatnonzero(
        (paq_data["data"][idx_2p][:-1] < threshold_volts) & 
        (paq_data["data"][idx_2p][1:] > threshold_volts)
    ) + 1
    
    print(f"Number of frames detected in 2P channel: {len(frame_count_2p)}")
    
    if len(frame_count_2p) <= tiff_frames:
        print("PAQ file does not need to be trimmed.")
        return paq_path
    
    # Find the index of the last frame we need
    last_frame_idx = frame_count_2p[tiff_frames - 1]
    print(f"Index of last frame to keep: {last_frame_idx}")
    
    print("Trimming PAQ data...")
    # Trim PAQ data to the calculated number of frames
    adjusted_data = paq_data["data"][:, :last_frame_idx]
    
    # Create new PAQ file
    if output_paq_path is None:
        base, ext = os.path.splitext(paq_path)
        output_paq_path = f"{base}_adjusted{ext}"
    
    print(f"Creating new PAQ file: {output_paq_path}")
    # Write the new PAQ file
    with open(output_paq_path, 'wb') as f:
        # Write sampling rate
        np.array(paq_data["rate"], dtype='>f').tofile(f)
        
        # Write number of channels
        np.array(len(paq_data["chan_names"]), dtype='>f').tofile(f)
        
        # Write channel names
        for chan_name in paq_data["chan_names"]:
            np.array(len(chan_name), dtype='>f').tofile(f)
            for char in chan_name:
                np.array(ord(char), dtype='>f').tofile(f)
        
        # Write hardware lines
        for hw_chan in paq_data["hw_chans"]:
            np.array(len(hw_chan), dtype='>f').tofile(f)
            for char in hw_chan:
                np.array(ord(char), dtype='>f').tofile(f)
        
        # Write units
        for unit in paq_data["units"]:
            np.array(len(unit), dtype='>f').tofile(f)
            for char in unit:
                np.array(ord(char), dtype='>f').tofile(f)
        
        # Write adjusted data
        adjusted_data.transpose().astype('>f').tofile(f)
    
    print(f"New PAQ file created with {last_frame_idx} frames: {output_paq_path}")
    return output_paq_path

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Adjusts a PAQ file to match the number of frames in a TIFF file')
    parser.add_argument('--info', action='store_true', help='Show PAQ file information')
    parser.add_argument('path1', nargs='?', help='Path to TIFF or PAQ file (depending on mode)')
    parser.add_argument('path2', nargs='?', help='Path to PAQ file (only in adjustment mode)')
    parser.add_argument('--output', help='Path to save the new PAQ file (optional)')
    
    args = parser.parse_args()
    
    if args.info:
        if not args.path1:
            print("Error: PAQ file path required when using --info")
            parser.print_help()
            sys.exit(1)
        show_paq_info(args.path1)
    else:
        if not args.path1 or not args.path2:
            print("Error: Both TIFF and PAQ file paths are required")
            parser.print_help()
            sys.exit(1)
        adjust_paq_to_tiff(args.path1, args.path2, args.output) 


def update_json_data(file_name, new_data, analysis_path=None):
    """
    Updates a JSON file with new data, checking for existing entries.
    
    Args:
        file_name: String with the JSON file name (e.g., "bias_data.json")
        new_data: Dictionary with the new data to add/update
        analysis_path: Optional string with the path to the analysis directory. 
                      If None, uses the default 'analysis' directory.
    
    Returns:
        bool: True if the update was successful, False if there was an error
    """
    import json
    import os
    
    try:
        # Determine JSON file path
        if analysis_path is None:
            analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
        json_path = os.path.join(analysis_path, file_name)
        
        # Load existing data if file exists
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
        
        # Check for existing entries
        existing_entries = []
        for key in new_data:
            if key in existing_data:
                if existing_data[key] == new_data[key]:
                    print(f"Warning: Entry '{key}' already exists with the same value '{new_data[key]}'")
                else:
                    print(f"Warning: Entry '{key}' exists with different value. Old: '{existing_data[key]}', New: '{new_data[key]}'")
                existing_entries.append(key)
        
        # Ask for confirmation if there are existing entries
        if existing_entries:
            response = input("Do you want to update the existing entries? (y/n): ")
            if response.lower() != 'y':
                print("Update cancelled")
                return False
        
        # Update with new data
        existing_data.update(new_data)
        
        # Save updated data
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f'File {file_name} successfully updated')
        return True
        
    except Exception as e:
        print(f'Error updating {file_name}: {str(e)}')
        return False
    
def add_ipsi_contra_columns(info, analysis_path):
    """
    Adds two new columns to the CSV files:
    1. ipsi_contra_bias: based on comparison between correctResponse and bias from bias_data.json
    2. ipsi_contra_recside: based on comparison between correctResponse and recording side from recside_data.json
    
    Args:
        info: Info object with recordingList
        analysis_path: String with the path to the analysis folder
    """
    import pandas as pd
    import json
    import os
    import glob
    
    # Load bias and recording side data
    bias_json_path = os.path.join(analysis_path, 'bias_data.json')
    recside_json_path = os.path.join(analysis_path, 'recside_data.json')
    
    with open(bias_json_path, 'r') as f:
        bias_data = json.load(f)
    with open(recside_json_path, 'r') as f:
        recside_data = json.load(f)
    
    # Process each session
    for ind, recordingDate in enumerate(info.recordingList.recordingDate):
        try:
            # Get CSV file path
            filenameCSV = info.recordingList.analysispathname[ind] + info.recordingList.sessionName[ind] + '_CorrectedeventTimes.csv'
            e_filenameCSV = [f for f in glob.glob(filenameCSV)]
            
            if len(e_filenameCSV) == 1:
                # Read CSV
                df = pd.read_csv(e_filenameCSV[0])
                
                # Get animal ID and session name
                animal_id = info.recordingList.sessionName[ind].split('_')[-1]
                session_name = info.recordingList.sessionName[ind]
                
                # Get bias and recording side
                animal_bias = bias_data.get(animal_id)
                session_side = recside_data.get(session_name)
                
                if animal_bias is None:
                    print(f"Warning: No bias data found for {animal_id}")
                    continue
                    
                if session_side is None:
                    print(f"Warning: No recording side data found for {session_name}")
                    continue
                
                # Add ipsi/contra columns based on bias
                df['ipsi_contra_bias'] = 'contra'  # default value
                if animal_bias == 'Left':
                    df.loc[df['correctResponse'] == 'Left', 'ipsi_contra_bias'] = 'ipsi'
                else:  # animal_bias == 'Right'
                    df.loc[df['correctResponse'] == 'Right', 'ipsi_contra_bias'] = 'ipsi'
                
                # Add ipsi/contra columns based on recording side
                df['ipsi_contra_recside'] = 'contra'  # default value
                if session_side == 'Left':
                    df.loc[df['correctResponse'] == 'Left', 'ipsi_contra_recside'] = 'ipsi'
                else:  # session_side == 'Right'
                    df.loc[df['correctResponse'] == 'Right', 'ipsi_contra_recside'] = 'ipsi'
                
                # Save updated CSV
                df.to_csv(e_filenameCSV[0], index=False)
                print(f"Updated CSV for session {session_name}")
                
        except Exception as e:
            print(f"Error processing session {ind}: {str(e)}")
            continue

def process_large_tiff_max_projections(tiff_path, frames_per_group=10000):
    """
    Process a large TIFF file and display maximum projections every N frames.
    
    Args:
        tiff_path (str): Path to the large TIFF file
        frames_per_group (int): Number of frames to group for each maximum projection
    """
    import tifffile
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # Get TIFF file information
    with tifffile.TiffFile(tiff_path) as tif:
        total_frames = len(tif.pages)
        print(f"Total frames in file: {total_frames}")
        
        # Calculate number of groups
        n_groups = total_frames // frames_per_group
        if total_frames % frames_per_group != 0:
            n_groups += 1
        
        # Process each group
        for group_idx in tqdm(range(n_groups), desc="Processing groups"):
            start_frame = group_idx * frames_per_group
            end_frame = min((group_idx + 1) * frames_per_group, total_frames)
            
            # Read frames from current group
            frames = []
            for frame_idx in range(start_frame, end_frame):
                frame = tif.pages[frame_idx].asarray()
                frames.append(frame)
            
            # Convert to numpy array and calculate maximum projection
            frames_array = np.stack(frames)
            max_proj = np.max(frames_array, axis=0)
            
            # Display maximum projection
            plt.figure(figsize=(10, 10))
            plt.imshow(max_proj, cmap='gray')
            plt.title(f'Maximum projection group {group_idx+1}/{n_groups}')
            plt.colorbar()
            plt.show()
            
            # Free memory
            del frames
            del frames_array
            del max_proj
    
    print(f"Processing completed. {n_groups} maximum projections displayed.")

def filter_responsive_neurons(dff_traces, pre_frames, post_frames, significance_level=0.05):
    from scipy import stats
    
    # Convert frames to seconds (assuming 30 fps)
    fps = 30
    pre_sec = 1  # 1 second before
    post_sec = 2  # 2 seconds after
    
    pre_window = int(pre_sec * fps)
    post_window = int(post_sec * fps)
    
    responsive_neurons = {}
    
    for condition, traces in dff_traces.items():
        if traces is None:
            responsive_neurons[condition] = None
            continue
            
        n_neurons = traces.shape[0]
        responsive = np.zeros(n_neurons, dtype=bool)
        
        for neuron in range(n_neurons):
            # Pre-event activity
            pre_activity = traces[neuron, pre_frames-pre_window:pre_frames]
            # Post-event activity
            post_activity = traces[neuron, pre_frames:pre_frames+post_window]
            
            # Student's t-test
            t_stat, p_val = stats.ttest_ind(post_activity, pre_activity)
            
            # Neuron responds if p < alpha 
            responsive[neuron] = (p_val < significance_level) 
            # and post activity > pre activity
            # and (np.mean(post_activity) > np.mean(pre_activity))
        
        responsive_neurons[condition] = responsive
        
    return responsive_neurons

def plot_dff_mean_by_contrast(session_path, save_path=None, use_responsive_only=True):
    """
    Plot dff mean by contrast for a session.
    
    Args:
        session_path: Path to the session directory
        save_path: Path where to save the figure. If None, uses session_path
        use_responsive_only: If True, uses data from responsive_neurons folder,
                           if False, uses data from all_neurons folder
    """
    # Determine subfolder based on use_responsive_only
    subfolder = 'responsive_neurons' if use_responsive_only else 'all_neurons'
    subfolder_path = os.path.join(session_path, subfolder)
    
    # Load data from the appropriate subfolder
    pickle_path = os.path.join(subfolder_path, 'imaging-dff_mean_zscored.pkl')
    with open(pickle_path, 'rb') as f:
        dff_mean_reward_zscored, dff_mean_stimuli_zscored, dff_mean_choice_zscored = pickle.load(f)
    
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(18, 5))
    contrasts = [-0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5]
    contrast_labels = [str(c) for c in contrasts]

    # Helper to get data for each contrast
    def get_data(dff_dict):
        data = []
        for contrast in contrasts:
            if contrast == 0:
                condition = 'zero contrast'
            elif contrast < 0:
                condition = f'contra {abs(contrast)}'
            else:
                condition = f'ipsi {contrast}'
            if condition in dff_dict and dff_dict[condition] is not None:
                for val in dff_dict[condition]:
                    data.append({'contrast': contrast, 'value': val})
        return pd.DataFrame(data)

    # Get all means for global y-axis
    all_means = []
    for dff_dict in [dff_mean_stimuli_zscored, dff_mean_choice_zscored, dff_mean_reward_zscored]:
        df = get_data(dff_dict)
        means = df.groupby('contrast')['value'].mean()
        all_means.extend(means.values)
    min_y = min(all_means) - 0.05
    max_y = max(all_means) + 0.05

    # Plot for each type
    for i, (title, dff_dict) in enumerate(zip(
        ['Stimulus Response', 'Choice Response', 'Reward Response'],
        [dff_mean_stimuli_zscored, dff_mean_choice_zscored, dff_mean_reward_zscored]
    )):
        plt.subplot(1, 3, i+1)
        df = get_data(dff_dict)
        sns.stripplot(x='contrast', y='value', data=df, order=contrasts, color='gray', size=4, alpha=0.6, jitter=True)
        sns.pointplot(
            x='contrast', y='value', data=df, order=contrasts, color='royalblue',
            capsize=0.15, err_kws={'linewidth': 2}, markers='o', linestyles='-'
        )
        plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5)  # x=4 is contrast==0
        plt.xlabel('Total Contrast')
        plt.ylabel('Mean Activity (z-score)')
        plt.title(title)
        plt.xticks(ticks=range(len(contrasts)), labels=contrast_labels, rotation=45)
        plt.ylim(min_y, max_y)
        plt.tight_layout()

    # If no save_path, use subfolder_path
    if save_path is None:
        save_path = subfolder_path
        
    # Save the figure in the specified path
    plt.savefig(os.path.join(save_path, 'dff_mean_by_contrast.png'))
    plt.close()