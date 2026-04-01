# This code  has plotting functions
from multiprocessing.util import info

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import statsmodels.api as sm
from scipy import stats
import main_funcs as mfun
import pickle
import os
from scipy.stats import ttest_ind
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import json



def set_figure():
    from matplotlib import rcParams
        # set the plotting values
    rcParams['figure.figsize'] = [12, 12]
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    rcParams['axes.spines.right']  = False
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.left']   = True
    rcParams['axes.spines.bottom'] = True

    params = {'axes.labelsize': 'large',
            'axes.titlesize':'large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large',
            'legend.fontsize': 'large'}
    
    rcParams.update(params)

def lineplot_withSEM_pupil (data, colorInd, label, axis):
    #lineplot_matrix(data=pupil_arr[session.outcome=='hit'], x_axis=x_axis, color=COLORS[0], label='hit')
    x_axis = np.linspace(-2, 6, data.shape[0])
    color =  sns.color_palette("bright")
    color  = [  (1,0,0), (0,0,0),color[2]]
    df = pd.DataFrame(data).melt()
    df['Time (sec)'] = np.tile(x_axis, data.shape[1])
    sns.lineplot(x='Time (sec)', y='value', data=df, color=color[colorInd],
                label=label,  ax = axis)
    #ylim_min = np.floor(np.min(np.nanmean(data,1)))*1.5
    #ylim_max = np.ceil(np.max(np.nanmean(data,1)))*1.5
    plt.ylabel('Pupil radius change (cm) ')
      
def lineplot_withSEM (data, colorInd, label, axis):
    #lineplot_matrix(data=pupil_arr[session.outcome=='hit'], x_axis=x_axis, color=COLORS[0], label='hit')
    x_axis = np.linspace(-2, 6, data.shape[0])
    color =  sns.color_palette("Paired")
    color  = [ color[5], (0,0,0),color[2]]
    df = pd.DataFrame(data).melt()
    df['Time (seconds)'] = np.tile(x_axis, data.shape[1])

    sns.lineplot(x='Time (seconds)', y='value', data=df, color=color[colorInd],
                label=label,  ax = axis)
    #ylim_min = np.floor(np.min(np.nanmean(data,1)))*1.5
    #ylim_max = np.ceil(np.max(np.nanmean(data,1)))*1.5
    ylim_min = np.floor(np.nanmin(np.nanmean(data,1)))*10
    ylim_max = np.ceil (np.nanmax(np.nanmean(data,1)))*10
    #if np.isnan(ylim_min): ylim_min = -1
    #if np.isnan(ylim_max): ylim_max = 5
    ylength = np.absolute(ylim_max - ylim_min)
    xlength = 0.25
    #add rectangle to plot
    ax = axis
    ax.add_patch (Rectangle ((0, ylim_min), xlength, ylength, alpha = 1, facecolor="grey",zorder=10))
    plt.ylabel('DFF')
    #ax.set_ylim( ymin =ylim_min, ymax = ylim_max)
    
def save_figure(name,base_path):
    plt.savefig(os.path.join(base_path, f'{name}.png'), 
                bbox_inches='tight', transparent=False)
   # plt.savefig(os.path.join(base_path, f'{name}.svg'), 
   #             bbox_inches='tight', transparent=True)

def save_figureAll(name,base_path):
    plt.savefig(os.path.join(base_path, f'{name}.png'), 
                bbox_inches='tight', transparent=False)
    plt.savefig(os.path.join(base_path, f'{name}.svg'), 
               bbox_inches='tight', transparent=True)

def lineplot_sessions(dffTrace_mean, analysis_params, colormap,
                    duration, zscoreRun, savefigname, savefigpath, baseline_subtract=None, 
                    title=None, axes=None, cellSelection = None, visualParams = None) :

    """
    Plots session mean traces for given analysis parameters and saves the figure.
    Optionally applies baseline subtraction and z-scoring.
    If 'title' is provided, it will be set as the plot title.
    """
    color = sns.color_palette(colormap, len(analysis_params))
    if visualParams is not None:
        fRate_imaging =  visualParams['fps']  # Hz
        pre_stim_sec = 2  # seconds before stimulus
        total_time = 8  # total time window (-2 to 6 seconds)
        step =  visualParams['fps']
    else:
        fRate_imaging = 30 # Hz
        pre_stim_sec = 2  # seconds before stimulus
        total_time = 8  # total time window (-2 to 6 seconds)
        step =30

    sessionsData = {}

    for indx, params in enumerate(analysis_params):
        array = dffTrace_mean[params]
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))

            # Apply baseline subtraction if specified
            if baseline_subtract is not None:
                # Calculate baseline from the specified time window
                # Assuming baseline_subtract is in seconds and we need to convert to frames
                # The time window is typically -2 to 6 seconds, so we need to map this

                # Convert baseline window from seconds to frame indices
                baseline_start_frame = int((baseline_subtract[0] + pre_stim_sec) * fRate_imaging)
                baseline_end_frame = int((baseline_subtract[1] + pre_stim_sec) * fRate_imaging)

                # Ensure indices are within bounds
                baseline_start_frame = max(0, baseline_start_frame)
                baseline_end_frame = min(array.shape[1], baseline_end_frame)

                if baseline_end_frame > baseline_start_frame:
                    # Calculate baseline mean for each cell
                    baseline_mean = np.nanmean(array[:, baseline_start_frame:baseline_end_frame], axis=1, keepdims=True)
                    # Subtract baseline
                    array = array - baseline_mean

            if zscoreRun:
                sessionsData[indx] = zscore(array, axis=1)
            else:
                sessionsData[indx] = array
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        xaxis_length = int(duration[0]) * fRate_imaging
    if axes is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    else:
        ax = axes
        
    for idx, sessionData in enumerate(sessionsData):
        plot_data = sessionsData[idx]
        if type(plot_data) != type(None):
            if cellSelection is not None:
                plot_data = plot_data[cellSelection,:]
            x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype=int)
            xticks = np.arange(0, len(x_labels), step)
            xticklabels = x_labels[::step]
            df = pd.DataFrame(plot_data).melt()
            # Smooth the data using lowess method from statsmodels
            x = df['variable']
            y = df['value']
            lowess_smoothed = sm.nonparametric.lowess(y, x, frac=0.1)
            sns.lineplot(x=x, y=y, data=df, color=color[idx],
                              label=analysis_params[idx], ax=ax)
            ax.axvline(x=pre_stim_sec*fRate_imaging, color='k', linestyle='--')
            ax.set_xticks(ticks=xticks)
            ax.set_xticklabels(xticklabels)
            #ax.set_ylim(bottom=np.nanmedian(plot_data)-np.nanstd(plot_data)*3, top=np.nanmedian(plot_data)+np.nanstd(plot_data)*3)
            ax.set_xlim(fRate_imaging, pre_stim_sec*fRate_imaging + xaxis_length)
            ax.set_xlabel('Time (sec)')
            if zscoreRun:
                ax.set_ylabel('DFF(zscore)')
            else:
                if baseline_subtract is not None:
                    ax.set_ylabel('DFF (baseline subtracted)')
                else:
                    ax.set_ylabel('DFF')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Set the title if provided
    if title is not None:
        ax.set_title(title)
    if axes is None:
        save_figure(savefigname, savefigpath)

def heatmap_sessions(dffTrace_mean,analysis_params, colormap,
                       selectedSession, duration, savefigname, savefigpath, axes = None,
                        params= None ) :
   
    if params == None: ## Parameters
        fps = 30 # for x ticks
        fRate = 1000/fps
        pre_frames    = 2000.0# in ms
        pre_frames    = int(np.ceil(pre_frames/fRate))
        post_frames   = 6000.0 # in ms
        post_frames   = int(np.ceil(post_frames/fRate))
        analysisWindowDur = 500 # in ms
        analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))
    else:
        fps = params['fps'] # for x ticks
        fRate =params['fRate']
        pre_frames    = params['pre_frames']# in ms
        post_frames   = params['post_frames'] # in ms
        analysisWindowDur = params['analysisWindowDur'] # in ms
    
    sessionsData = [None] * len(analysis_params)

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        # if there is only one cell/trace
        elif array.shape[0] == 1:
            nCell = array.shape[2] # This is trial number this time
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))
            sessionsData[indx] =  zscore(array, axis=1)
            YaxisLabelSt = 'Trial #'
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))
            sessionsData[indx]= zscore(array, axis = 1)
            YaxisLabelSt = 'Cell #'
        
    ymaxValue = 2.5
    yminValue = -2.5
    
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        yaxis_length = int(duration[0])*fps
    

        grid_ratio = [1 for _ in range(len(analysis_params))]
        grid_ratio.append(0.05) # for the colorbar

        if axes == None: 
            fig, axes = plt.subplots(nrows=1, ncols=len(analysis_params)+1, figsize=((len(analysis_params)+1)*5,10 ), 
                                gridspec_kw={'width_ratios': grid_ratio})
        else:
            axes = axes

        for idx, sessionData in enumerate(sessionsData):
            plot_data = sessionsData[idx]
            if type(plot_data) != type(None):
                if selectedSession == 'WithinSession':
                    sortedInd = np.array(np.nanmean(plot_data[:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]
                else:
                    # Make sure the selected index exists
                    if selectedSession < len(sessionsData) and sessionsData[selectedSession] is not None:
                        sortedInd = np.array(np.nanmean(sessionsData[selectedSession][:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]
                    else:
                        sortedInd = np.array(np.nanmean(plot_data[:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]

                # Make sure the selected indices are within the limits
                sortedInd = sortedInd[sortedInd < plot_data.shape[0]]
                plot_data = plot_data[sortedInd]
                
                nT = plot_data.shape[1]
                t = (np.arange(nT) - pre_frames) / fps   # time vector in seconds
                tick_times = np.arange(-2, 7, 1)         # -2,-1,0,1,...,6
                xticks = np.round(tick_times * fps + pre_frames).astype(int)
                mask = (xticks >= 0) & (xticks < nT)
                xticks = xticks[mask]
                xticklabels = tick_times[mask]

                # create ylabels - lets find out the number of cells
                nCell = plot_data.shape[0]
                step = 10 if nCell <= 500 else 100
                yticks = np.arange(0, nCell, step)          # positions (0-based, row indices)
                yticklabs = (yticks).astype(int)        # labels shown (1-based cell numbers)
                
                axes[idx] = sns.heatmap(plot_data, vmin = yminValue, vmax = ymaxValue, cbar = False, yticklabels = False,cmap = colormap, ax = axes[idx])
                axes[idx].axvline(x=pre_frames, color='w', linewidth = 3)
                axes[idx].set_yticks(yticks + 0.5)
                axes[idx].set_yticklabels(yticklabs)
                if idx == 0:
                    axes[idx].set_ylabel(YaxisLabelSt)
                else:
                    axes[idx].set_ylabel('')
                    axes[idx].tick_params(axis='y', left=False, labelleft=False)
                axes[idx].set_xlim(fps,pre_frames+yaxis_length)
                axes[idx].set_xticks (ticks = xticks, labels= xticklabels)
                axes[idx].set_xticklabels(xticklabels, rotation=0)
                axes[idx].set_xlabel('Time (sec)')
                axes[idx].set_title(analysis_params[idx]+ '\n' + YaxisLabelSt + ': ' + str(plot_data.shape[0]))

        # Create a dummy heatmap solely for the color bar
        cax = axes[-1].inset_axes([0.2, 0.2, 0.6, 0.6])
        sns.heatmap(np.zeros((1, 1)), ax=cax, cbar=True, cbar_ax=axes[-1], cmap=colormap, cbar_kws={'label': 'DFF','shrink': 0.5})

        save_figure(savefigname,savefigpath)

def histogram_sessions(dffTrace_mean,analysis_params,colormap, zscoreRun, savefigname, savefigpath ) :
    
    ## Parameters
    fRate = 1000/30
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    post_frames   = 6000.0 # in ms
    post_frames   = int(np.ceil(post_frames/fRate))
    analysisWindowDur = 500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))
    binrange = np.arange(-2.5, 2.6, 0.20)
    color =  sns.color_palette(colormap, len(analysis_params))
    sessionsData = []

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]
        if np.array_equal(array, np.array(None)):
            sessionsData.append(None)
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))
            if zscoreRun:
                array = zscore(array, axis = 1)
            sessionsData.append( np.nanmean(array[:, (pre_frames): (pre_frames + analysisWindowDur)],1) )
    xtickbinrange = np.arange(-2.5, 2.6, 0.5)
    if  type(sessionsData[-1])!= type(None):
    #~np.array_equal(sessionsData[-1], np.array(None)): 
        # Set up the subplots with shared x-axis
        fig, axs = plt.subplots((len(sessionsData)+1), 1, figsize=(6, (3*(len(sessionsData)+1))))
        plt.subplots_adjust(hspace=0.2)
        # Calculate the maximum frequency for all histograms

        max_freq = max([np.histogram(data, bins=binrange)[0].max() for data in sessionsData if data is not None])
        mean_all = []
        var_all  =[]
        for indx, data in enumerate(sessionsData):
            if data is not None:
                # Create the histogram using sns.histplot()
                mean_all.append(np.mean(data))
                var_all.append(np.var(data))
                sns.histplot(data, kde=True, bins=binrange, color= color[indx], ax=axs[indx])
                axs[indx].set_ylim(0, np.round(max_freq*1.1))
                if indx >0:
                    h, p =  stats.ks_2samp(data, datapre)
                    axs[indx].text(0.95, 0.95, f'KS values: {h:.5f}\np: {p:.4f}', transform=axs[indx].transAxes,
                    va='top', ha='right', color='black')
                elif indx ==0:
                    datapre = data
                # Add a dashed line at the mean position
                axs[indx].axvline(x=np.nanmean(data), color='black', linestyle='--')
            

            # Add labels and title
            axs[indx].set_ylabel(analysis_params[indx] + '\n numCells')
            axs[indx].set_xlim(-2.5, 2.5)

            # Set x-axis label for the bottom subplot
            axs[indx].set_xticks(xtickbinrange)
            axs[indx].set_xticklabels(xtickbinrange)
        axs[indx].set_xlabel('Average dFF') 
       
        # add mean plot
        # Create a new grid for the last row of subplots
        last_row_axes = plt.subplot2grid((4, 5), (3, 0), colspan=2)

        last_row_axes.plot(mean_all, '+', 
                           color='black', markersize = 10)
        last_row_axes.axhline(y=0, color='black', linestyle='--')
        last_row_axes.set_xticks( np.arange(0, 3, 1))
        last_row_axes.set_xticklabels(analysis_params, fontsize ='small')
        last_row_axes.set_ylabel('Mean') 

        # add mean plot
        # Create a new grid for the last row of subplots
        last_row_axes = plt.subplot2grid((4, 5), (3, 3), colspan=2)
        
        last_row_axes.plot(var_all, '+', 
                           color='black', markersize = 10)
        last_row_axes.axhline(y=0, color='black', linestyle='--')
        last_row_axes.set_xticks( np.arange(0, 3, 1))
        last_row_axes.set_xticklabels(analysis_params, fontsize ='small')
        last_row_axes.set_ylabel('Variance') 
        
        save_figureAll(savefigname,savefigpath)

def scatter_sessions(dffTrace_mean1, dffTrace_mean2, analysis_params,
                      label, colormap, zscoreRun, savefigname, savefigpath ) :
    
    ## Parameters
    fRate = 1000/30
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    post_frames   = 6000.0 # in ms
    post_frames   = int(np.ceil(post_frames/fRate))
    analysisWindowDur = 500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))
    color =  sns.color_palette(colormap, len(analysis_params))
    sessionsData1 = []
    sessionsData2 = []

    for indx, params in enumerate(analysis_params) :
        array1 = dffTrace_mean1[params]
        array2 = dffTrace_mean2[params]
        if np.array_equal(array1, np.array(None)):
            sessionsData1.append( None)
            sessionsData2.append( None)
        elif np.array_equal(array2, np.array(None)):
            sessionsData1.append( None)
            sessionsData2.append( None)
        else:
            nCell = array1.shape[0]
            analysis_window = array1.shape[1]
            array1 = np.reshape(array1, (nCell, analysis_window))
            array2 = np.reshape(array2, (nCell, analysis_window))
            if zscoreRun:
                array1 = zscore(array1, axis = 1)
                array2 = zscore(array2, axis = 1)

            sessionsData1.append( np.nanmean(array1[:, (pre_frames): (pre_frames + analysisWindowDur)],1))
            sessionsData2.append( np.nanmean(array2[:, (pre_frames): (pre_frames + analysisWindowDur)],1))
    max_freq1 = max([np.max(np.abs(data)) for data in sessionsData1 if data is not None])
    max_freq2 = max([np.max(np.abs(data)) for data in sessionsData2 if data is not None])
    max_value = max(max_freq1,max_freq2)

    # Create the plot
    fig, axs = plt.subplots(len(sessionsData1), 1, figsize=(6, 6*len(sessionsData1)))
    for indx, data in enumerate(sessionsData1):
        if data is not None:
           if sessionsData2[indx] is not None:
            sns.scatterplot(x = data, y =sessionsData2[indx] , color= color[indx], ax = axs[indx])
    
        axs[indx].plot([max_value*-1, max_value], [max_value*-1, max_value], color='black', linestyle='--')
        axs[indx].set_xlim(max_value*-1, max_value)
        axs[indx].set_ylim(max_value*-1, max_value)
        axs[indx].set_ylabel(analysis_params[indx] + '\n' +label[1])
        axs[indx].set_xlabel(label[0])

    save_figureAll(savefigname,savefigpath)

def plot_combined_psychometric(recordingList, save_fileName=None, return_df=False):
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Dictionary to store probabilities for each contrast across sessions
    contrast_probs = {}
    # For the output DataFrame
    session_diffs = []
    # For each session in recordingList
    for ind in range(len(recordingList)):
        try:
            session = recordingList.sessionName[ind]
            #print(f"\nProcessing session: {session}")
            
            # Construct CSV path
            csv_path = os.path.join(recordingList.analysispathname[ind], 
                                  f"{session}_CorrectedeventTimes.csv")
            
            if not os.path.exists(csv_path):
                print(f"CSV file not found at: {csv_path}")
                continue
            
            b = pd.read_csv(csv_path)
            good_trials = (b['repeatNumber'] == 1) & (b['choice'] != 'NoGo')
            c_diff = b['contrastRight'] - b['contrastLeft']
            unique_contrasts = np.unique(c_diff)
            right_probs = []
            right_probs_ci = []
            prob_0 = []
            for contrast in unique_contrasts:
                trials = (c_diff == contrast) & good_trials
                if np.sum(trials) > 0:
                    right_choices = (b['choice'][trials] == 'Right')
                    prob = np.mean(right_choices)
                    # Binomial confidence interval
                    ci = stats.binom.interval(0.95, np.sum(trials), prob)
                    right_probs.append(prob)
                    right_probs_ci.append([prob - ci[0]/np.sum(trials), 
                                         ci[1]/np.sum(trials) - prob])
                    # Store probability for combined plot
                    if contrast not in contrast_probs:
                        contrast_probs[contrast] = []
                    contrast_probs[contrast].append(prob)
                    # For the DataFrame
                    if contrast == 0:
                        prob_0.append(prob)
            # Save in the list if valid data
            if len(prob_0) > 0:
                diff = np.mean(prob_0) - 0.5  # difference to chance
                session_diffs.append({'sessionName': session, 'diff_0_vs_chance': diff})
            # Plot individual session data
            ax1.errorbar(unique_contrasts, right_probs, 
                        yerr=np.array(right_probs_ci).T,
                        fmt='o-', label=f'Session {session}', alpha=0.7)
        except Exception as e:
            print(f"Error processing session {session}: {str(e)}")
            continue
    # Calculate mean and standard error for each contrast
    EXCLUDE = np.array([-0.0625, 0.0625], dtype=float)
    keys_sorted = sorted(contrast_probs.keys(), key=float)
    contrasts = [k for k in keys_sorted
             if not np.any(np.isclose(float(k), EXCLUDE, rtol=0, atol=1e-12))]
    print(contrasts)
    mean_probs = [np.mean(contrast_probs[c]) for c in contrasts]
    sem_probs = [np.std(contrast_probs[c], ddof=1) / np.sqrt(len(contrast_probs[c])) 
                for c in contrasts]
    # Plot combined mean with standard error
    ax2.plot(contrasts, mean_probs, 'k-', label='Mean across sessions', linewidth=2 ,
             marker='o')
    ax2.fill_between(contrasts, 
                    np.array(mean_probs) - np.array(sem_probs),
                    np.array(mean_probs) + np.array(sem_probs),
                    alpha=0.2, color='k')
    # Customize individual sessions plot
    ax1.set_xlabel('Contrast')
    ax1.set_ylabel('P(Right) & 95% CI')
    ax1.set_title('Individual Sessions')
    ax1.grid(True, alpha=0.2)
    # put a legend right side of plot
    #ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim([-0.6, 0.6])
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.2)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.2)
    # Customize combined plot
    ax2.set_xlabel('Contrast')
    ax2.set_ylabel('P(Right)')
    ax2.set_title('Mean & SEM (All sessions n = {})'.format(len(recordingList)))
    ax2.grid(True, alpha=0.3)
    #ax2.legend()
    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.2)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.2)
    plt.tight_layout()
    if save_fileName is not None: # Not so good practice to have this as a default
        full_save_path = os.path.join(save_fileName)
        plt.savefig(full_save_path)
        plt.close()
        print(f"Figure saved at: {full_save_path}")
    else:
        plt.show()
    if return_df:
        df = pd.DataFrame(session_diffs)
        return df

def getContrastBehData(recordingList):
    EXCLUDE = np.array([-0.0625, 0.0625], dtype=float)
    session_data   = []
    contrast_probs = defaultdict(list)

    for ind in range(len(recordingList)):
            _, _,b = mfun.clean_session_behavior(recordingList.analysispathname[ind], 
                                       recordingList.sessionName[ind], recordingList.path[ind])

            good   = (b['repeatNumber'] == 1) & (b['choice'] != 'NoGo')
            c_diff = b['contrastRight'] - b['contrastLeft']

            contrasts, probs = [], []
            for contrast in sorted(c_diff.unique()):
                if np.any(np.isclose(float(contrast), EXCLUDE, atol=1e-12)):
                    continue
                trials = good & (c_diff == contrast)
                if trials.sum() > 0:
                    prob = (b.loc[trials, 'choice'] == 'Right').mean()
                    contrasts.append(float(contrast))
                    probs.append(prob)
                    contrast_probs[float(contrast)].append(prob)

            if contrasts:
                session_data.append({'name': recordingList.sessionName[ind], 'contrasts': contrasts, 'probs': probs})

    return {'session_data': session_data, 'contrast_probs': dict(contrast_probs)}

def plotContrastBeh(recordingList, axis=None, individualSession=True):
    """
    Plot psychometric contrast-response curve on a given axis.
    Mean ± SEM across sessions, with optional individual session traces.

    Parameters
    ----------
    recordingList     : DataFrame-like with sessionName, analysispathname columns
    axis              : matplotlib Axes. Creates a new figure if None.
    individualSession : bool, overlay individual session traces if True.
    """
    data           = getContrastBehData(recordingList)
    session_data   = data['session_data']
    contrast_probs = data['contrast_probs']

    ax = axis if axis is not None else plt.subplots(figsize=(5, 5))[1]

    # ---- individual sessions ----
    if individualSession:
        for s in session_data:
            ax.plot(s['contrasts'], s['probs'],
                    color='gray', linewidth=1, alpha=0.3, marker='o', markersize=3)

    # ---- mean ± SEM ----
    sorted_contrasts = sorted(contrast_probs)
    mean_probs = [np.mean(contrast_probs[c]) for c in sorted_contrasts]
    sem_probs  = [np.std(contrast_probs[c], ddof=1) / np.sqrt(len(contrast_probs[c]))
                  for c in sorted_contrasts]

    ax.plot(sorted_contrasts, mean_probs, 'k-', linewidth=2, marker='o', markersize=5,
            label=f'Mean (n={len(session_data)})')
    ax.fill_between(sorted_contrasts,
                    np.array(mean_probs) - np.array(sem_probs),
                    np.array(mean_probs) + np.array(sem_probs),
                    alpha=0.2, color='k')

    ax.axhline(0.5, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0,   color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Contrast')
    ax.set_ylabel('P(Correct)')
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.2)

    return ax

def create_FOV_withSelectedCells(isCell, ops, s2p_path, analysispathname, savename,
                                 stat = None):
    prob_threshold = 0
    cell_indices = np.where((isCell[:,1] > prob_threshold))[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Show mean image
    ax.imshow(ops['meanImg'], cmap='binary_r')

    # Load stat.npy to get cell coordinates
    if stat is None:
        stat = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)

    # Generate random colors for each cell
    colors = np.random.rand(len(cell_indices), 3)

    # Draw ROIs of filtered cells
    for idx, cell_number in enumerate(cell_indices):
        stat_cell = stat[cell_number]
        ypix = [stat_cell['ypix'][i] for i in range(len(stat_cell['ypix'])) if not stat_cell['overlap'][i]]
        xpix = [stat_cell['xpix'][i] for i in range(len(stat_cell['xpix'])) if not stat_cell['overlap'][i]]
        ax.plot(xpix, ypix, '.', markersize=1, alpha=0.7, color=colors[idx])

    ax.set_title(f"Detected cells (prob > {prob_threshold}): {len(cell_indices)}")
    ax.axis('off')
    plt.show()
    fig.savefig(
        os.path.join(analysispathname, savename + "_MeanImG.png"),
        dpi=300,          # higher resolution
        bbox_inches='tight',  # trim whitespace
        pad_inches=0.1,       # small padding
        facecolor='white'     # ensure background matches
    )

    # Also show enhanced image (meanImgE) which might be clearer
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(ops['meanImgE'], cmap='binary_r')

    # Draw ROIs of filtered cells
    for idx, cell_number in enumerate(cell_indices):
        stat_cell = stat[cell_number]
        ypix = [stat_cell['ypix'][i] for i in range(len(stat_cell['ypix'])) if not stat_cell['overlap'][i]]
        xpix = [stat_cell['xpix'][i] for i in range(len(stat_cell['xpix'])) if not stat_cell['overlap'][i]]
        ax.plot(xpix, ypix, '.', markersize=1, alpha=0.7, color=colors[idx])

    ax.set_title(f"Detected cells (prob > {prob_threshold}): {len(cell_indices)}")
    ax.axis('off')
    # save this image in the analysis folder
    fig.savefig(
        os.path.join(analysispathname, savename + f"_FOV_withSelectedCells_{prob_threshold}.png"),
        dpi=300,          # higher resolution
        bbox_inches='tight',  # trim whitespace
        pad_inches=0.1,       # small padding
        facecolor='white'     # ensure background matches
    )
    #close the plot
    plt.close(fig)

def plotMeanDffByContrast(
    recordingList,
    event_type='stimulus',
    time_window=[0.1, 0.8],
    save_path=None,
    title=None,
    subfolder='responsive_neurons',
    dffTrace_mean_dict=None,
    use_zscored=True,
    baseline_window=[-0.2, 0],
    contrasts_rewarded=None,
    contrast_values=None,
    response_type='all',
    axis=None,
    align_to_hemisphere=False,
    session_normalization='zero_center'):   # None, 'zero_center', 'zero_center_scale'


    # -- PARAMS --
    fRate_imaging = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    start_frame = int((time_window[0] + pre_stim_sec) * fRate_imaging)
    end_frame = int((time_window[1] + pre_stim_sec) * fRate_imaging)

    # ---- defaults ----
    if contrasts_rewarded is None:
        contrasts_rewarded = [
            '0 Rewarded',
            '0.0625 Rewarded',
            '0.125 Rewarded',
            '0.25 Rewarded',
            '0.5 Rewarded'
        ]
    if contrast_values is None:
        contrast_values = [0.0, 0.0625, 0.125, 0.25, 0.5]

    # ---- figure ----
    if axis is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axis

    # Use aligned numeric contrast as key
    contrast_data = defaultdict(list)
    session_points = {}

    # ---- process sessions ----
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] != 1:
            continue

        animal_id = recordingList.animalID[ind]
        session_name = recordingList.sessionName[ind]
        pathname = recordingList.analysispathname[ind]
        subfolder_path = os.path.join(pathname, subfolder)

        try:
            # hemisphere
            if align_to_hemisphere:
                hemisphere = recordingList.recordingHemisphere[ind]
            else:
                hemisphere = None

            # load data
            if use_zscored:
                pkl_name = 'imaging-dffTrace_mean_zscored.pkl'
            else:
                pkl_name = 'imaging-dffTrace_mean.pkl'

            with open(os.path.join(subfolder_path, pkl_name), 'rb') as f:
                dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)

            if event_type.lower() == 'stimulus':
                dffTrace_mean = dffTrace_mean_stimuli
            elif event_type.lower() == 'choice':
                dffTrace_mean = dffTrace_mean_choice
            elif event_type.lower() == 'reward':
                dffTrace_mean = dffTrace_mean_reward
            else:
                raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")

            # collect this session first
            session_means_by_contrast = {}

            for i, contrast in enumerate(contrasts_rewarded):
                if contrast not in dffTrace_mean or dffTrace_mean[contrast] is None:
                    continue

                session_trace = dffTrace_mean[contrast]
                if session_trace.shape[1] <= end_frame:
                    continue

                # ---- baseline indices ----
                if baseline_window is not None:
                    baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                    baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                else:
                    baseline_start_idx = np.argmin(np.abs(time_axis - (-0.2)))
                    baseline_end_idx = np.argmin(np.abs(time_axis - 0))

                if baseline_end_idx <= baseline_start_idx:
                    baseline_end_idx = baseline_start_idx + 1

                # ---- baseline subtraction per cell ----
                baseline = np.nanmean(
                    session_trace[:, baseline_start_idx:baseline_end_idx],
                    axis=1,
                    keepdims=True
                )
                dff_bs = session_trace - baseline

                # ---- choose cells ----
                response_metric = np.nanmean(dff_bs[:, start_frame:end_frame], axis=1)

                if response_type == 'excited':
                    cellIndex = response_metric > 0
                elif response_type == 'inhibited':
                    cellIndex = response_metric < 0
                elif response_type == 'all':
                    cellIndex = np.ones(session_trace.shape[0], dtype=bool)
                else:
                    raise ValueError("response_type must be 'excited', 'inhibited', or 'all'")

                if np.sum(cellIndex) == 0:
                    continue

                dff_bs = dff_bs[cellIndex, :]

                # mean across selected cells and time window
                mean_dff = np.nanmean(dff_bs[:, start_frame:end_frame])

                raw_contrast = contrast_values[i]
                if align_to_hemisphere:
                    plot_contrast = align_contrast_to_hemisphere(raw_contrast, hemisphere)
                else:
                    plot_contrast = raw_contrast

                session_means_by_contrast[plot_contrast] = mean_dff

            # ---- session normalization ----
            if len(session_means_by_contrast) == 0:
                continue

            norm_vals = session_means_by_contrast.copy()

            if session_normalization in ['zero_center', 'zero_center_scale']:
                if 0.0 not in session_means_by_contrast:
                    # skip session if no zero contrast available for zero-centering
                    continue

                zero_val = session_means_by_contrast[0.0]
                centered = {k: v - zero_val for k, v in session_means_by_contrast.items()}

                if session_normalization == 'zero_center':
                    norm_vals = centered

                elif session_normalization == 'zero_center_scale':
                    scale = np.max(np.abs(list(centered.values())))
                    if scale == 0 or np.isnan(scale):
                        scale = 1.0
                    norm_vals = {k: v / scale for k, v in centered.items()}

            elif session_normalization is None:
                norm_vals = session_means_by_contrast.copy()

            else:
                raise ValueError("session_normalization must be None, 'zero_center', or 'zero_center_scale'")

            # ---- store session points ----
            sorted_contrasts = sorted(norm_vals.keys())
            sorted_means = [norm_vals[c] for c in sorted_contrasts]

            if len(sorted_contrasts) > 1:
                session_points[session_name] = (sorted_contrasts, sorted_means)

            for c in sorted_contrasts:
                contrast_data[c].append({
                    'animal': animal_id,
                    'session': session_name,
                    'mean_dff': norm_vals[c],
                    'contrast': c
                })

        except Exception as e:
            print(f"Error processing session {session_name}: {str(e)}")
            continue

    # ---- plot individual session lines ----
    for session_name, (xvals, yvals) in session_points.items():
        ax.plot(
            xvals, yvals,
            linestyle='--',
            color='gray',
            alpha=0.5,
            linewidth=1,
            zorder=1
        )

    # ---- scatter points ----
    unique_contrasts = sorted(contrast_data.keys())
    if len(unique_contrasts) > 0:
        plot_colors = sns.color_palette('viridis', len(unique_contrasts))

        for i, c in enumerate(unique_contrasts):
            df_contrast = pd.DataFrame(contrast_data[c])
            ax.scatter(
                df_contrast['contrast'],
                df_contrast['mean_dff'],
                color=plot_colors[i],
                alpha=0.5,
                s=40,
                zorder=2
            )

    # ---- mean + sem ----
    means = []
    sems = []
    x_vals = []

    for c in unique_contrasts:
        vals = [d['mean_dff'] for d in contrast_data[c]]
        if len(vals) > 0:
            means.append(np.nanmean(vals))
            sems.append(np.nanstd(vals) / np.sqrt(len(vals)))
            x_vals.append(c)

    if len(x_vals) > 0:
        ax.plot(
            x_vals, means,
            color='k',
            linewidth=2,
            label='Mean across sessions',
            zorder=5
        )
        ax.errorbar(
            x_vals, means, yerr=sems,
            fmt='o',
            color='k',
            elinewidth=2,
            capsize=5,
            alpha=0.8,
            markersize=6,
            zorder=6
        )

    # ---- labels ----
    if align_to_hemisphere:
        ax.set_xlabel('Aligned contrast (ipsi ← 0 → contra)')
    else:
        ax.set_xlabel('Contrast')

    if session_normalization == 'zero_center':
        ax.set_ylabel('Average dF/F \n(session zero-centered)')
    elif session_normalization == 'zero_center_scale':
        ax.set_ylabel('Average dF/F \n(zero-centered, scaled)')
    else:
        ax.set_ylabel('Average dF/F')

    if title is None:
        title = f'AnalysisWin: {time_window[0]}-{time_window[1]}s post-{event_type}'
    ax.set_title(title, y=1.04, fontsize=14)

    ax.set_xscale('linear')
    if len(unique_contrasts) > 0:
        ax.set_xticks(unique_contrasts)
        ax.set_xticklabels([str(c) for c in unique_contrasts])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)


    # Optional y limits
    ax.set_ylim(-0.1, 0.6)

    if save_path is not None and axis is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax

def align_contrast_to_hemisphere(raw_contrast, hemisphere):
    # negative = ipsi, positive = contra
    if hemisphere == 'Left':
        return raw_contrast
    elif hemisphere == 'Right':
        return -raw_contrast
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")

def createMeanDffByContrastData(
    recordingList,
    event_type='stimulus',
    time_window=[0.1, 0.8],
    subfolder='all_neurons',
    use_zscored=False,
    baseline_window=[-0.2, 0],
    response_type='all',
    align_to_hemisphere=False,
    session_normalization='zero_center',   # None, 'zero_center', 'zero_center_scale'
):


    # ---- validate once ----
    event_lower = event_type.lower()
    if event_lower not in ('stimulus', 'choice', 'reward'):
        raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")
    if session_normalization not in (None, 'zero_center', 'zero_center_scale'):
        raise ValueError("session_normalization must be None, 'zero_center', or 'zero_center_scale'")
    if response_type not in ('excited', 'inhibited', 'all'):
        raise ValueError("response_type must be 'excited', 'inhibited', or 'all'")

    # ---- defaults ----
    contrasts_rewarded = ['-0.5 Rewarded','-0.25 Rewarded','-0.125 Rewarded', '0 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded']
    contrast_values    = [-0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5]


    # ---- time parameters (computed once) ----
    fRate_imaging, pre_stim_sec, total_time, _, _ = mfun.get_imagingExtractionParams()
    n_frames  = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # ---- baseline indices (computed once) ----
    bl_start = (baseline_window[0] if baseline_window is not None else -0.2)
    bl_end   = (baseline_window[1] if baseline_window is not None else 0.0)
    bsl_idx  = np.argmin(np.abs(time_axis - bl_start))
    bsl_end  = max(np.argmin(np.abs(time_axis - bl_end)), bsl_idx + 1)

    pkl_name = 'imaging-dffTrace_zscoredAll.pkl' if use_zscored else 'imaging-dffTraceAll.pkl'
    pkl_key  = {'stimulus': 1, 'choice': 2, 'reward': 0}[event_lower]

    # ---- helper: baseline-subtract + windowed mean -> (n_cells,) ----
    # dffTrace_mean is 2D (cells × time) — already averaged across trials
    def _metric(trace, sf, ef):
        with np.errstate(all='ignore'):
            _sf = int(np.mean(sf)) if hasattr(sf, '__len__') else sf
            _ef = int(np.mean(ef)) if hasattr(ef, '__len__') else ef
            bl  = np.nanmean(trace[:, bsl_idx:bsl_end], axis=1, keepdims=True)
            dff = trace - bl
            return np.nanmean(dff[:, _sf:_ef], axis=1)     # (n_cells,)

    # ---- outputs ----
    contrast_data  = defaultdict(list)
    session_points = {}

    # ---- process sessions ----
    for ind in range(len(recordingList.recordingDate)):
        try:
            if recordingList.imagingDataExtracted[ind] != 1:
                continue

            sf, ef, idx_keep, behData = mfun.get_startFramePerTrial(recordingList.analysispathname[ind],
                                            recordingList.sessionName[ind], recordingList.path[ind], event_type, 
                                            exclude = None, behOut= True)

            if time_window is not None:
                sf = int((time_window[0] + pre_stim_sec) * fRate_imaging)
                ef = int((time_window[1] + pre_stim_sec) * fRate_imaging)

            animal_id    = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            hemisphere   = recordingList.recordingHemisphere[ind] if align_to_hemisphere else None
            subfolder_path = os.path.join(recordingList.analysispathname[ind], subfolder)

            with open(os.path.join(subfolder_path, pkl_name), 'rb') as f:
                dffTrace = pickle.load(f)[pkl_key]

            # ---- average trials per contrast ----
            tTypesName, tTypes = mfun.build_trial_types(behData,
                            recordingList.imagingTiffFileNames[ind], idx_keep)
            dffTrace_mean = {}
            nCells, nTime, _ = dffTrace['All Trials'].shape

            for i, contrast_name in enumerate(contrasts_rewarded):
                if contrast_name not in tTypesName:
                    continue
                selectedTrials = tTypes[tTypesName.index(contrast_name)]
                if np.sum(selectedTrials) > 2:
                    dffTrace_mean[contrast_name] = np.mean(
                        dffTrace['All Trials'][:, :, selectedTrials], axis=2)  # (cells × time x trials)
                else:
                    dffTrace_mean[contrast_name] = np.full((nCells, nTime), np.nan)

            # ---- available contrasts for this session ----
            available = [
                (i, c) for i, c in enumerate(contrasts_rewarded)
                if c in dffTrace_mean and not np.all(np.isnan(dffTrace_mean[c]))
            ]
            if not available:
                print(f"Skipping {session_name}: no valid contrast data")
                continue

            # ---- cellIndex from highest contrast (computed once per session) ----
            all_metrics = []
            for _, contrast in available:
                all_metrics.append(_metric(dffTrace_mean[contrast], sf, ef))
            all_metrics = np.vstack(all_metrics)  # contrasts x cells
            ref_metric = np.nanmean(all_metrics, axis=0)

            if response_type == 'excited':
                cellIndex = ref_metric > 0
            elif response_type == 'inhibited':
                cellIndex = ref_metric < 0
            else:
                cellIndex = np.ones(len(ref_metric), dtype=bool)

            if not np.any(cellIndex):
                continue

            # ---- per-contrast mean response ----
            session_means_by_contrast = {}
            for i, contrast in available:
                cell_responses = _metric(dffTrace_mean[contrast], sf, ef)[cellIndex]
                raw_c = contrast_values[i]
                plot_c = align_contrast_to_hemisphere(raw_c, hemisphere) if align_to_hemisphere else raw_c
                session_means_by_contrast[plot_c] = float(np.nanmean(cell_responses))

            session_mean = np.nanmean(list(session_means_by_contrast.values()))
            session_std = np.nanstd(list(session_means_by_contrast.values()))
            session_std = max(session_std, 1e-10)

            session_means_by_contrast = { contrast: (value - session_mean) / session_std
                            for contrast, value in session_means_by_contrast.items()}

            if not session_means_by_contrast:
                continue

            # ---- session normalization ----
            if session_normalization in ('zero_center', 'zero_center_scale'):
                if 0.0 not in session_means_by_contrast:
                    continue
                zero_val = session_means_by_contrast[0.0]
                centered = {k: v - zero_val for k, v in session_means_by_contrast.items()}
                if session_normalization == 'zero_center':
                    norm_vals = centered
                else:
                    scale = np.max(np.abs(list(centered.values())))
                    norm_vals = {k: v / max(scale, 1e-10) for k, v in centered.items()}
            else:
                norm_vals = session_means_by_contrast

            # ---- store ----
            sorted_contrasts = sorted(norm_vals)
            if len(sorted_contrasts) > 1:
                session_points[session_name] = (sorted_contrasts, [norm_vals[c] for c in sorted_contrasts])

            for c in sorted_contrasts:
                contrast_data[c].append({
                    'animal': animal_id, 'session': session_name,
                    'hemisphere': hemisphere, 'mean_dff': norm_vals[c], 'contrast': c
                })
        except Exception as e:
            print(f"Error processing session {session_name}: {str(e)}")

    # ---- dataframes ----
    summary_df = pd.DataFrame([d for c in sorted(contrast_data) for d in contrast_data[c]])

    if summary_df.empty:
        mean_df = pd.DataFrame(columns=['contrast', 'mean', 'sem', 'n_sessions'])
    else:
        mean_df = (
            summary_df.groupby('contrast')['mean_dff']
            .agg(mean='mean', sem=lambda x: x.std() / np.sqrt(len(x)), n_sessions='count')
            .reset_index()
        )

    return {
        'contrast_data': contrast_data,
        'session_points': session_points,
        'summary_df': summary_df,
        'mean_df': mean_df,
        'meta': {
            'event_type': event_type, 'time_window': time_window,
            'baseline_window': baseline_window, 'response_type': response_type,
            'align_to_hemisphere': align_to_hemisphere,
            'session_normalization': session_normalization,
            'contrasts_rewarded': contrasts_rewarded, 'contrast_values': contrast_values,
        }
    }

def plotMeanDffByContrastFromData(
    data_dict,
    axis=None,
    title=None,
    save_path=None,
    ylim=(-0.1, 0.6),
    show_scatter=True,
    show_session_lines=True,
    show_mean=True,
    grid_alpha=0.3,
):
    """
    Plot mean dF/F by contrast from precomputed data created by
    createMeanDffByContrastData().
    """

    contrast_data = data_dict['contrast_data']
    session_points = data_dict['session_points']
    mean_df = data_dict['mean_df']
    meta = data_dict['meta']

    if axis is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        ax = axis

    unique_contrasts = sorted(contrast_data.keys())

    # ---- session lines ----
    if show_session_lines:
        for session_name, (xvals, yvals) in session_points.items():
            ax.plot(
                xvals, yvals,
                linestyle='--',
                color='gray',
                alpha=0.5,
                linewidth=1,
                zorder=1
            )

    # ---- scatter ----
    if show_scatter and len(unique_contrasts) > 0:
        plot_colors = sns.color_palette('viridis', len(unique_contrasts))
        for i, c in enumerate(unique_contrasts):
            df_contrast = pd.DataFrame(contrast_data[c])
            ax.scatter(
                df_contrast['contrast'],
                df_contrast['mean_dff'],
                color=plot_colors[i],
                alpha=0.5,
                s=40,
                zorder=2
            )

    # ---- mean + sem ----
    if show_mean and not mean_df.empty:
        ax.plot(
            mean_df['contrast'],
            mean_df['mean'],
            color='k',
            linewidth=2,
            label='Mean across sessions',
            zorder=5
        )
        ax.errorbar(
            mean_df['contrast'],
            mean_df['mean'],
            yerr=mean_df['sem'],
            fmt='o',
            color='k',
            elinewidth=2,
            capsize=5,
            alpha=0.8,
            markersize=6,
            zorder=6
        )

    # ---- labels ----
    if meta['align_to_hemisphere']:
        ax.set_xlabel('Aligned contrast (ipsi ← 0 → contra)')
    else:
        ax.set_xlabel('Contrast')

    if meta['session_normalization'] == 'zero_center':
        ax.set_ylabel('Average dF/F \n(session zero-centered)')
    elif meta['session_normalization'] == 'zero_center_scale':
        ax.set_ylabel('Average dF/F \n(zero-centered, scaled)')
    else:
        ax.set_ylabel('Average dF/F')

    if title is None:
        title = f"AnalysisWin: {meta['time_window'][0]}-{meta['time_window'][1]}s post-{meta['event_type']}"
    ax.set_title(title, y=1.04, fontsize=14)

    ax.set_xscale('linear')
    if len(unique_contrasts) > 0:
        ax.set_xticks(unique_contrasts)
        ax.set_xticklabels([str(c) for c in unique_contrasts])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=grid_alpha)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if save_path is not None and axis is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax

def contrastHemisphereDifference (data_out):
    df = data_out['summary_df']
    contrast_levels = sorted(abs(c) for c in df['contrast'].unique())

    rows = []
    for c in contrast_levels:

        if c == 0:
            # keep zero contrast as it is
            zero_df = df[df['contrast'] == 0][['session', 'mean_dff']].copy()
            zero_df['contrast'] = 0
            zero_df['hemiDiff'] = zero_df['mean_dff']
            rows.append(zero_df)

        else:
            left = df[df['contrast'] == -c][['session', 'mean_dff', 'hemisphere']]
            right = df[df['contrast'] == c][['session', 'mean_dff', 'hemisphere']]
            merged = pd.merge(left, right, on=['session', 'hemisphere'], suffixes=('_left', '_right'))
            merged['contrast'] = c
            merged['hemiDiff'] = merged['mean_dff_right'] - merged['mean_dff_left']
            rows.append(merged)

    contrast_diff_df = pd .concat(rows, ignore_index=True)

    contrast_summary = (
        contrast_diff_df
        .groupby('contrast')['hemiDiff']
        .agg(['mean','sem','count']))
    return contrast_summary

def create_glm_dataframe(data_out, recordingList, bias_col='zeroBias'):
    """
    Merge session-level mean_dff-by-contrast data with recording metadata.
    
    Parameters
    ----------
    data_out : dict
        Output from createMeanDffByContrastData()
    recordingList : pd.DataFrame
        Must contain sessionName, recordingHemisphere, and bias_col
    bias_col : str
        Column name in recordingList containing session bias
    
    Returns
    -------
    glm_df : pd.DataFrame
    """
    df = data_out['summary_df'].copy()
    meta_cols = ['sessionName', 'recordingHemisphere', 'PsychoFitBias', 
                 'performance','zeroBias', 'zeroBias_PsychoFitBias_sameSign', 'animalID' ]
    meta = recordingList[meta_cols].copy().drop_duplicates()

    glm_df = df.merge(
        meta,
        left_on='session',
        right_on='sessionName',
        how='left'
    )
    
    glm_df = glm_df.drop(columns=['sessionName'])

    # clean hemisphere coding
    glm_df['hemisphere'] = glm_df['recordingHemisphere'].map({
        'Left': -1,
        'Right': 1
    })

    glm_df = glm_df.dropna(subset=['mean_dff', 'contrast', bias_col, 'hemisphere'])

    # optional centered variables (recommended for interactions)
    glm_df['contrast_c'] = glm_df['contrast'] - glm_df['contrast'].mean()
    glm_df['bias_c'] = glm_df[bias_col] - glm_df[bias_col].mean()
    glm_df['alignment'] = glm_df['zeroBias_PsychoFitBias_sameSign']
 
    return glm_df