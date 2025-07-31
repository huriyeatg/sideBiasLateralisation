# This code  has plotting functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import statsmodels.api as sm
from scipy import stats
import pickle
import os
from scipy.stats import ttest_ind
import itertools
from collections import defaultdict


import os


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
                    duration, zscoreRun, savefigname, savefigpath, baseline_subtract=None, title=None):
    plt.figure(figsize=(5, 3))
    """
    Plots session mean traces for given analysis parameters and saves the figure.
    Optionally applies baseline subtraction and z-scoring.
    If 'title' is provided, it will be set as the plot title.
    """
    color = sns.color_palette(colormap, len(analysis_params))
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
                fRate_imaging = 30  # Hz
                pre_stim_sec = 2  # seconds before stimulus
                total_time = 8  # total time window (-2 to 6 seconds)

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
    step = 30  # for x ticks
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        xaxis_length = int(duration[0]) * 30
    plt.subplot(2, 2, 1)
    for idx, sessionData in enumerate(sessionsData):
        plot_data = sessionsData[idx]
        if type(plot_data) != type(None):
            x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype=int)
            xticks = np.arange(0, len(x_labels), step)
            xticklabels = x_labels[::step]
            df = pd.DataFrame(plot_data).melt()
            # Smooth the data using lowess method from statsmodels
            x = df['variable']
            y = df['value']
            lowess_smoothed = sm.nonparametric.lowess(y, x, frac=0.1)
            ax = sns.lineplot(x=x, y=y, data=df, color=color[idx],
                              label=analysis_params[idx])
            plt.axvline(x=60, color='k', linestyle='--')
            plt.xticks(ticks=xticks, labels=xticklabels)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.xlim(30, 60 + xaxis_length)
            plt.xlabel('Time (sec)')
            if zscoreRun:
                plt.ylabel('DFF(zscore)')
            else:
                if baseline_subtract is not None:
                    plt.ylabel('DFF (baseline subtracted)')
                else:
                    plt.ylabel('DFF')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Set the title if provided
    if title is not None:
        plt.title(title)
    save_figureAll(savefigname, savefigpath)

def heatmap_sessions(dffTrace_mean,analysis_params, colormap,
                       selectedSession, duration, savefigname, savefigpath ) :
    ## Parameters
    fRate = 1000/30
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    post_frames   = 6000.0 # in ms
    post_frames   = int(np.ceil(post_frames/fRate))
    analysisWindowDur = 500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))
    sessionsData = [None] * len(analysis_params)

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))
            sessionsData[indx]= zscore(array, axis = 1)
        
    ymaxValue = 2.5
    yminValue = -2.5
    step = 30 # for x ticks
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        yaxis_length = int(duration[0])*30
    

        grid_ratio = [1 for _ in range(len(analysis_params))]
        grid_ratio.append(0.05) # for the colorbar
        fig, axes = plt.subplots(nrows=1, ncols=len(analysis_params)+1, figsize=((len(analysis_params)+1)*2.5, 1.5), 
                                gridspec_kw={'width_ratios': grid_ratio})

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
                
                x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype = int)
                xticks = np.arange(0, len(x_labels), step)
                xticklabels = x_labels[::step]
                
                ax = sns.heatmap(plot_data, vmin = yminValue, vmax = ymaxValue, cbar = False, yticklabels = False,cmap = colormap, ax = axes[idx])
                ax.axvline(x=pre_frames, color='w', linewidth = 3)
                ax.set_xticks (ticks = xticks, labels= xticklabels)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_xlim(30,pre_frames+yaxis_length)
                ax.set_xlabel('Time (sec)')
                ax.set_title(analysis_params[idx])

        # Create a color bar for all heatmaps next to the last subplot
        # Hide the y-axis label for the dummy heatmap
        axes[-1].set_yticks([])
        # Create a dummy heatmap solely for the color bar
        cax = axes[-1].inset_axes([0.2, 0.2, 0.6, 0.6])
        sns.heatmap(np.zeros((1, 1)), ax=cax, cbar=True, cbar_ax=axes[-1], cmap=colormap, cbar_kws={'label': 'DFF','shrink': 0.5})
        axes[0].set_ylabel('Cells')
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

def plot_combined_psychometric(info, save_path=None, return_df=False):
    """
    Creates two plots:
    1. Individual right turn probability vs contrast for each session
    2. Combined mean and standard error across all sessions
    Optionally returns a DataFrame with sessionName and the difference between mean(0 contrast) and chance (0.5).
    Args:
        info: Info object containing recordingList with session information
        save_path: Path where to save the figure. If None, only shows the figure
        return_df: If True, returns a DataFrame with sessionName and diff (mean(0 contrast) - 0.5)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import pandas as pd
    import os
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    
    # Dictionary to store probabilities for each contrast across sessions
    contrast_probs = {}
    # For the output DataFrame
    session_diffs = []
    
    # For each session in recordingList
    for ind in range(len(info.recordingList)):
        try:
            session = info.recordingList.sessionName[ind]
            print(f"\nProcessing session: {session}")
            
            # Construct CSV path
            csv_path = os.path.join(info.recordingList.analysispathname[ind], 
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
    contrasts = sorted(contrast_probs.keys())
    mean_probs = [np.mean(contrast_probs[c]) for c in contrasts]
    sem_probs = [np.std(contrast_probs[c], ddof=1) / np.sqrt(len(contrast_probs[c])) 
                for c in contrasts]
    # Plot combined mean with standard error
    ax2.plot(contrasts, mean_probs, 'b-', label='Mean across sessions', linewidth=2)
    ax2.fill_between(contrasts, 
                    np.array(mean_probs) - np.array(sem_probs),
                    np.array(mean_probs) + np.array(sem_probs),
                    alpha=0.2, color='b')
    # Customize individual sessions plot
    ax1.set_xlabel('Contrast')
    ax1.set_ylabel('p(Right) & 95% CI')
    ax1.set_title('Individual Sessions')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    # Customize combined plot
    ax2.set_xlabel('Contrast')
    ax2.set_ylabel('p(Right) & SEM')
    ax2.set_title('Combined Mean')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        animal_id = info.recordingList.animalID[0]
        full_save_path = os.path.join(save_path, f'{animal_id}_combined_psychometric.png')
        plt.savefig(full_save_path)
        plt.close()
        print(f"Figure saved at: {full_save_path}")
    else:
        plt.show()
    if return_df:
        df = pd.DataFrame(session_diffs)
        return df

def plot_combined_response_time(info, analysis_path, save_path=None):
    """
    Creates four plots:
    1. Individual response time vs contrast for each session
    2. Combined mean and standard error across all sessions
    3. Individual response time vs contrast for each session, separated by ipsi/contra
    4. Combined mean and standard error across all sessions, separated by ipsi/contra
    
    Args:
        info: Info object containing recordingList with session information
        analysis_path: String with the path to the analysis folder
        save_path: Path where to save the figure. If None, uses default path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import pandas as pd
    import os
    import json
    
    # Load bias data from JSON
    json_path = os.path.join(analysis_path, 'bias_data.json')
    with open(json_path, 'r') as f:
        bias_data = json.load(f)
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5, 6))
    
    # Dictionary to store response times for each contrast across sessions
    contrast_times = {}
    contrast_times_ipsi = {}
    contrast_times_contra = {}
    
    # Define markers for different sessions
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # For each session in recordingList
    for ind in range(len(info.recordingList)):
        try:
            session = info.recordingList.sessionName[ind]
            
            # Get animal ID from session name
            animal_id = session.split('_')[-1]
            
            # Get bias for this animal
            if animal_id not in bias_data:
                continue
                
            animal_bias = bias_data[animal_id]
            
            # Construct CSV path
            csv_path = os.path.join(info.recordingList.analysispathname[ind], 
                                  f"{session}_CorrectedeventTimes.csv")
            
            # Read CSV file
            b = pd.read_csv(csv_path)
            
            # Filter good trials
            good_trials = (b['repeatNumber'] == 1) & (b['choice'] != 'NoGo')
            
            # Calculate contrast difference
            c_diff = b['contrastRight'] - b['contrastLeft']
            
            # Calculate response time
            response_time = b['choiceCompleteTime'] - b['stimulusOnsetTime']
            
            # Determine which trials are ipsi/contra based on bias
            if animal_bias == 'Left':
                ipsi_trials = b['contrastLeft'] > b['contrastRight']
                contra_trials = b['contrastRight'] > b['contrastLeft']
            else:  # animal_bias == 'Right'
                ipsi_trials = b['contrastRight'] > b['contrastLeft']
                contra_trials = b['contrastLeft'] > b['contrastRight']
            
            # Calculate mean response time for each contrast
            unique_contrasts = np.unique(c_diff)
            
            # Initialize arrays for all contrasts
            mean_times = np.full_like(unique_contrasts, np.nan, dtype=float)
            sem_times = np.full_like(unique_contrasts, np.nan, dtype=float)
            mean_times_ipsi = np.full_like(unique_contrasts, np.nan, dtype=float)
            sem_times_ipsi = np.full_like(unique_contrasts, np.nan, dtype=float)
            mean_times_contra = np.full_like(unique_contrasts, np.nan, dtype=float)
            sem_times_contra = np.full_like(unique_contrasts, np.nan, dtype=float)
            
            for i, contrast in enumerate(unique_contrasts):
                # All trials
                trials = (c_diff == contrast) & good_trials
                if np.sum(trials) > 0:
                    times = response_time[trials]
                    mean_times[i] = np.mean(times)
                    sem_times[i] = np.std(times, ddof=1) / np.sqrt(len(times))
                    
                    # Store times for combined plot
                    if contrast not in contrast_times:
                        contrast_times[contrast] = []
                    contrast_times[contrast].extend(times)
                
                # Ipsi trials
                trials = (c_diff == contrast) & good_trials & ipsi_trials
                if np.sum(trials) > 0:
                    times = response_time[trials]
                    mean_times_ipsi[i] = np.mean(times)
                    sem_times_ipsi[i] = np.std(times, ddof=1) / np.sqrt(len(times))
                    
                    # Store times for combined plot
                    if contrast not in contrast_times_ipsi:
                        contrast_times_ipsi[contrast] = []
                    contrast_times_ipsi[contrast].extend(times)
                
                # Contra trials
                trials = (c_diff == contrast) & good_trials & contra_trials
                if np.sum(trials) > 0:
                    times = response_time[trials]
                    mean_times_contra[i] = np.mean(times)
                    sem_times_contra[i] = np.std(times, ddof=1) / np.sqrt(len(times))
                    
                    # Store times for combined plot
                    if contrast not in contrast_times_contra:
                        contrast_times_contra[contrast] = []
                    contrast_times_contra[contrast].extend(times)
            
            # Plot individual session data (all trials)
            ax1.errorbar(unique_contrasts, mean_times, 
                        yerr=sem_times,
                        fmt=f'{markers[ind]}-', label=f'Session {session}', alpha=0.7)
            
            # Plot individual session data (ipsi/contra)
            ax3.errorbar(unique_contrasts, mean_times_ipsi, 
                        yerr=sem_times_ipsi,
                        fmt=f'{markers[ind]}-', color='blue', label=f'Session {session} Ipsi', alpha=0.7)
            ax3.errorbar(unique_contrasts, mean_times_contra, 
                        yerr=sem_times_contra,
                        fmt=f'{markers[ind]}-', color='red', label=f'Session {session} Contra', alpha=0.7)
            
        except Exception as e:
            print(f"Error processing session {session}: {str(e)}")
            continue
    
    # Calculate mean and standard error for each contrast
    contrasts = sorted(set(contrast_times.keys()))
    contrasts_ipsi_contra = sorted(set(contrast_times_ipsi.keys()) | set(contrast_times_contra.keys()))
    
    # Initialize arrays for all contrasts
    mean_times = np.full_like(contrasts, np.nan, dtype=float)
    sem_times = np.full_like(contrasts, np.nan, dtype=float)
    mean_times_ipsi = np.full_like(contrasts_ipsi_contra, np.nan, dtype=float)
    sem_times_ipsi = np.full_like(contrasts_ipsi_contra, np.nan, dtype=float)
    mean_times_contra = np.full_like(contrasts_ipsi_contra, np.nan, dtype=float)
    sem_times_contra = np.full_like(contrasts_ipsi_contra, np.nan, dtype=float)
    
    # All trials
    for i, c in enumerate(contrasts):
        if contrast_times[c]:
            mean_times[i] = np.mean(contrast_times[c])
            sem_times[i] = np.std(contrast_times[c], ddof=1) / np.sqrt(len(contrast_times[c]))
    
    # Ipsi/Contra
    for i, c in enumerate(contrasts_ipsi_contra):
        if c in contrast_times_ipsi and contrast_times_ipsi[c]:
            mean_times_ipsi[i] = np.mean(contrast_times_ipsi[c])
            sem_times_ipsi[i] = np.std(contrast_times_ipsi[c], ddof=1) / np.sqrt(len(contrast_times_ipsi[c]))
        
        if c in contrast_times_contra and contrast_times_contra[c]:
            mean_times_contra[i] = np.mean(contrast_times_contra[c])
            sem_times_contra[i] = np.std(contrast_times_contra[c], ddof=1) / np.sqrt(len(contrast_times_contra[c]))
    
    # Plot combined mean with standard error (all trials)
    ax2.plot(contrasts, mean_times, 'k-', label='Mean across sessions', linewidth=2)
    ax2.fill_between(contrasts, 
                    mean_times - sem_times,
                    mean_times + sem_times,
                    alpha=0.2, color='k')
    
    # Plot combined mean with standard error (ipsi/contra)
    ax4.plot(contrasts_ipsi_contra, mean_times_ipsi, 'b-', label='Ipsilateral', linewidth=2)
    ax4.fill_between(contrasts_ipsi_contra, 
                    mean_times_ipsi - sem_times_ipsi,
                    mean_times_ipsi + sem_times_ipsi,
                    alpha=0.2, color='b')
    
    ax4.plot(contrasts_ipsi_contra, mean_times_contra, 'r-', label='Contralateral', linewidth=2)
    ax4.fill_between(contrasts_ipsi_contra, 
                    mean_times_contra - sem_times_contra,
                    mean_times_contra + sem_times_contra,
                    alpha=0.2, color='r')
    
    # Customize plots
    for ax, title in zip([ax1, ax2, ax3, ax4], 
                        ['Individual Sessions', 'Combined Mean', 
                         'Individual Sessions (Ipsi/Contra)', 'Combined Mean (Ipsi/Contra)']):
        ax.set_xlabel('Contrast')
        ax.set_ylabel('Response Time (s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labels
            ax.legend()
        ax.set_xlim([-1, 1])
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Set default save path if none provided
    if save_path is None:
        save_path = r"C:\Users\Lak Lab\Documents\Github\sideBiasLateralisation\analysis\figs"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get animal ID from first session
    animal_id = info.recordingList.animalID[0]
    
    # Create filename with animal ID
    full_save_path = os.path.join(save_path, f'{animal_id}_combined_response_time.png')
    
    plt.savefig(full_save_path)
    plt.close()
    
    print(f"Figure saved at: {full_save_path}")



def plot_combined_dff_mean_by_contrast(info, zscoreRun=True, use_responsive_only=True, save_path=None):
    """
    Creates a combined plot of dff mean by contrast across multiple sessions.
    
    Args:
        info: Info object containing recordingList with session information
        zscoreRun: If True, uses z-scored data from imaging-dff_mean_zscored.pkl, 
                  if False, uses raw data from imaging-dff_mean.pkl
        use_responsive_only: If True, uses data from responsive_neurons folder,
                           if False, uses data from all_neurons folder
        save_path: Path where to save the figure. If None, uses default path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import seaborn as sns
    from scipy import stats
    
    # Set up the figure
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(9, 2.5))
    
    # Define contrasts
    contrasts = [-0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5]
    contrast_labels = [str(c) for c in contrasts]
    
    # Dictionary to store data for each type (stimuli, choice, reward)
    all_data = {
        'Stimulus Response': [],
        'Choice Response': [],
        'Reward Response': []
    }
    
    # For each session in recordingList
    for ind in range(len(info.recordingList)):
        try:
            session = info.recordingList.sessionName[ind]
            print(f"\nProcessing session: {session}")
            
            # Determine subfolder based on use_responsive_only
            subfolder = 'responsive_neurons' if use_responsive_only else 'all_neurons'
            
            # Build pickle file path based on zscoreRun
            if zscoreRun:
                pickle_path = os.path.join(info.recordingList.analysispathname[ind], 
                                         subfolder,
                                         'imaging-dff_mean_zscored.pkl')
            else:
                pickle_path = os.path.join(info.recordingList.analysispathname[ind], 
                                         subfolder,
                                         'imaging-dff_mean.pkl')
            
            # Check if file exists
            if not os.path.exists(pickle_path):
                print(f"Pickle file not found at: {pickle_path}")
                continue
            
            # Read pickle file
            with open(pickle_path, 'rb') as f:
                dff_mean_reward, dff_mean_stimuli, dff_mean_choice = pickle.load(f)
            
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
            
            # Process each type
            for title, dff_dict in zip(
                ['Stimulus Response', 'Choice Response', 'Reward Response'],
                [dff_mean_stimuli, dff_mean_choice, dff_mean_reward]
            ):
                df = get_data(dff_dict)
                if not df.empty:
                    all_data[title].append(df)
            
        except Exception as e:
            print(f"Error processing session {session}: {str(e)}")
            continue
    
    # Plot for each type
    for i, (title, session_data) in enumerate(all_data.items()):
        plt.subplot(1, 3, i+1)
        
        if session_data:  # If we have data for this type
            # Combine all sessions
            combined_df = pd.concat(session_data, ignore_index=True)
            
            # Plot
            sns.stripplot(x='contrast', y='value', data=combined_df, 
                         order=contrasts, color='gray', size=4, alpha=0.6, jitter=True)
            sns.pointplot(
                x='contrast', y='value', data=combined_df, order=contrasts,
                color='royalblue', capsize=0.15, err_kws={'linewidth': 2},
                markers='o', linestyles='-'
            )
            
            plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5)  # x=4 is contrast==0
            plt.xlabel('Total Contrast')
            plt.ylabel('Mean Activity (z-score)' if zscoreRun else 'Mean Activity')
            plt.title(title)
            plt.xticks(ticks=range(len(contrasts)), labels=contrast_labels, rotation=45)
            
            # Set y-axis limits based on all data
            min_y = combined_df['value'].min() - 0.05
            max_y = combined_df['value'].max() + 0.05
            plt.ylim(min_y, max_y)
    
    plt.tight_layout()
    
    # Set default save path if none provided
    if save_path is None:
        save_path = r"C:\Users\Lak Lab\Documents\Github\sideBiasLateralisation\analysis\figs"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get animal ID from first session
    animal_id = info.recordingList.animalID[0]
    
    # Create filename with animal ID, data type and neuron type
    data_type = 'zscored' if zscoreRun else 'raw'
    neuron_type = 'responsive' if use_responsive_only else 'all'
    full_save_path = os.path.join(save_path, 
                                 f'{animal_id}_combined_dff_mean_by_contrast_{neuron_type}_{data_type}.png')
    
    plt.savefig(full_save_path)
    plt.close()
    
    print(f"Figure saved at: {full_save_path}")

def plot_combined_neural_activity(info, analysis_params, colormap='Set2', duration=[3], zscoreRun=True, 
                                save_path=None, use_responsive_only=True):
    """
    Creates a combined plot of average neural activity across multiple sessions.
    
    Args:
        info: Info object containing recordingList with session information
        analysis_params: List of parameters to analyze (e.g., ['Rewarded', 'Unrewarded'])
        colormap: Color map for the plot
        duration: Analysis duration in seconds
        zscoreRun: If True, uses z-scored data from imaging-dffTrace_mean_zscored.pkl, 
                  if False, uses raw data from imaging-dffTrace_mean.pkl
        save_path: Path where to save the figure. If None, uses default path
        use_responsive_only: If True, uses data from responsive_neurons folder,
                           if False, uses data from all_neurons folder
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import seaborn as sns
    from scipy import stats
    
    # Create figure
    plt.figure(figsize=(7.5, 3))
    
    # Dictionary to store data for each condition
    condition_data = {param: [] for param in analysis_params}
    
    # For each session in recordingList
    for ind in range(len(info.recordingList)):
        try:
            session = info.recordingList.sessionName[ind]
            print(f"\nProcessing session: {session}")
            
            # Determine subfolder based on use_responsive_only
            subfolder = 'responsive_neurons' if use_responsive_only else 'all_neurons'
            
            # Build pickle file path based on zscoreRun and use_responsive_only
            if zscoreRun:
                pickle_path = os.path.join(info.recordingList.analysispathname[ind], 
                                         subfolder,
                                         'imaging-dffTrace_mean_zscored.pkl')
            else:
                pickle_path = os.path.join(info.recordingList.analysispathname[ind], 
                                         subfolder,
                                         'imaging-dffTrace_mean.pkl')
            
            # Check if file exists
            if not os.path.exists(pickle_path):
                print(f"Pickle file not found at: {pickle_path}")
                continue
            
            # Read pickle file
            with open(pickle_path, 'rb') as f:
                dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
            
            # Select data type based on analysis parameters
            if any(param in ['Rewarded', 'Unrewarded'] for param in analysis_params):
                dffTrace_mean = dffTrace_mean_reward
            elif any(param in ['Left choices', 'Right choices'] for param in analysis_params):
                dffTrace_mean = dffTrace_mean_choice
            else:
                dffTrace_mean = dffTrace_mean_stimuli
            
            # Process each parameter
            for param in analysis_params:
                if param in dffTrace_mean and dffTrace_mean[param] is not None:
                    # Calculate cell average
                    mean_activity = np.mean(dffTrace_mean[param], axis=0)
                    condition_data[param].append(mean_activity)
            
        except Exception as e:
            print(f"Error processing session {session}: {str(e)}")
            continue
    
    # Calculate mean and standard error for each condition
    colors = sns.color_palette(colormap, len(analysis_params))
    
    for idx, param in enumerate(analysis_params):
        if condition_data[param]:
            # Convert to numpy array
            all_sessions = np.array(condition_data[param])
            
            # Calculate mean and standard error
            mean_activity = np.mean(all_sessions, axis=0)
            sem_activity = np.std(all_sessions, axis=0) / np.sqrt(len(all_sessions))
            
            # Create x axis
            x = np.linspace(-2, 6, len(mean_activity))
            
            # Plot
            plt.plot(x, mean_activity, color=colors[idx], label=param, linewidth=2)
            plt.fill_between(x, 
                           mean_activity - sem_activity,
                           mean_activity + sem_activity,
                           color=colors[idx], alpha=0.2)
    
    # Customize plot
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('DFF (z-score)' if zscoreRun else 'DFF')
    plt.title('Combined Average Neural Activity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adjust x-axis limits
    yaxis_length = int(duration[0]) * 30
    plt.xlim(-2, 6)
    
    plt.tight_layout()
    
    # Set default save path if none provided
    if save_path is None:
        save_path = r"C:\Users\Lak Lab\Documents\Github\sideBiasLateralisation\analysis\figs"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get animal ID from first session
    animal_id = info.recordingList.animalID[0]
    
    # Create filename with animal ID, data type and neuron type
    data_type = 'zscored' if zscoreRun else 'raw'
    neuron_type = 'responsive' if use_responsive_only else 'all'
    full_save_path = os.path.join(save_path, 
                                 f'{animal_id}_combined_neural_activity_{neuron_type}_{data_type}_{duration[0]}sec.png')
    plt.savefig(full_save_path)
    plt.close()
    
    print(f"Figure saved at: {full_save_path}")

def plot_single_session_response_time_histogram(session_path, session_name, save_path=None, bins=30, 
                                              exclude_zero_contrast=True, figsize=(6, 4)):
    """
    Creates a histogram of response times for a single session, excluding trials with contrast=0.
    
    Args:
        session_path: String with the path to the session folder
        session_name: String with the session name
        save_path: Path where to save the figure. If None, uses session_path
        bins: Number of bins for the histogram (default: 30)
        exclude_zero_contrast: Boolean to exclude trials with contrast=0 (default: True)
        figsize: Tuple with figure size (default: (12, 8))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    
    # Construct CSV path
    csv_path = os.path.join(session_path, f"{session_name}_CorrectedeventTimes.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Read CSV file
    b = pd.read_csv(csv_path)
    
    # Filter good trials
    good_trials = (b['repeatNumber'] == 1) & (b['choice'] != 'NoGo')
    
    # Calculate contrast difference
    c_diff = b['contrastRight'] - b['contrastLeft']
    
    # Calculate response time
    response_time = b['choiceCompleteTime'] - b['stimulusOnsetTime']
    
    # Filter out trials with contrast=0 if requested
    if exclude_zero_contrast:
        non_zero_trials = c_diff != 0
        good_trials = good_trials & non_zero_trials
    
    # Get response times for good trials
    valid_response_times = response_time[good_trials]
    
    if len(valid_response_times) == 0:
        print(f"Error: No valid trials found for session {session_name}")
        return
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Overall histogram of response times
    ax1.hist(valid_response_times, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Response Time (s)')
    ax1.set_ylabel('Number of Trials')
    ax1.set_title(f'Response Time Distribution - {session_name}')
    ax1.grid(True, alpha=0.3)
    
    # Add more x-axis ticks for better resolution
    x_min, x_max = ax1.get_xlim()
    x_ticks = np.linspace(x_min, x_max, 20)  # 20 ticks across the range
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{tick:.2f}' for tick in x_ticks], rotation=45, ha='right')
    
    # Add statistics
    mean_rt = np.mean(valid_response_times)
    median_rt = np.median(valid_response_times)
    std_rt = np.std(valid_response_times)
    ax1.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.3f}s')
    ax1.axvline(median_rt, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rt:.3f}s')
    ax1.legend()
    
    # Plot 2: Response time vs contrast
    unique_contrasts = np.unique(c_diff[good_trials])
    mean_times = []
    sem_times = []
    
    for contrast in unique_contrasts:
        trials = (c_diff == contrast) & good_trials
        times = response_time[trials]
        mean_times.append(np.mean(times))
        sem_times.append(np.std(times, ddof=1) / np.sqrt(len(times)))
    
    ax2.errorbar(unique_contrasts, mean_times, yerr=sem_times, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax2.set_xlabel('Contrast Difference (Right - Left)')
    ax2.set_ylabel('Mean Response Time (s)')
    ax2.set_title('Response Time vs Contrast')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: Separate histograms for positive and negative contrasts
    positive_trials = (c_diff > 0) & good_trials
    negative_trials = (c_diff < 0) & good_trials
    
    positive_times = response_time[positive_trials]
    negative_times = response_time[negative_trials]
    
    if len(positive_times) > 0:
        ax3.hist(positive_times, bins=bins, alpha=0.7, color='green', 
                label=f'Positive (n={len(positive_times)})', edgecolor='black')
    if len(negative_times) > 0:
        ax3.hist(negative_times, bins=bins, alpha=0.7, color='red', 
                label=f'Negative (n={len(negative_times)})', edgecolor='black')
    
    ax3.set_xlabel('Response Time (s)')
    ax3.set_ylabel('Number of Trials')
    ax3.set_title('Response Time by Contrast Sign')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add more x-axis ticks for better resolution
    x_min, x_max = ax3.get_xlim()
    x_ticks = np.linspace(x_min, x_max, 20)  # 20 ticks across the range
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([f'{tick:.2f}' for tick in x_ticks], rotation=45, ha='right')
    
    # Plot 4: Box plot of response times by contrast
    contrast_groups = []
    contrast_labels = []
    
    for contrast in unique_contrasts:
        trials = (c_diff == contrast) & good_trials
        times = response_time[trials]
        if len(times) > 0:
            contrast_groups.append(times)
            contrast_labels.append(f'{contrast:.3f}')
    
    if contrast_groups:
        bp = ax4.boxplot(contrast_groups, labels=contrast_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax4.set_xlabel('Contrast Difference')
        ax4.set_ylabel('Response Time (s)')
        ax4.set_title('Response Time Distribution by Contrast')
        ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot instead of showing it
    save_path = os.path.join(session_path, f'{session_name}_response_time_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Response time histogram saved to: {save_path}")
    
    return fig

def plot_individual_cell_traces_by_contrast(dffTrace_stimuli, session_name, save_path=None, 
                                           figsize=(7.5, 30), n_cols=3, time_window=[-2, 6], baseline_window=None,
                                           contrast_conditions=None, contrast_labels=None):
    """
    Plots the mean trace of each cell aligned to stimulus, separated by contrast using the viridis colormap.
    Supports baseline subtraction and custom contrast conditions/labels.

    Args:
        dffTrace_stimuli: dict, keys are contrast conditions, values are arrays (cell x time x trial)
        session_name: str, session name for the plot title
        save_path: str or None, directory to save the plot (if None, shows the plot)
        figsize: tuple, figure size
        n_cols: int, number of columns in the subplot grid
        time_window: list, [start_time, end_time] in seconds
        baseline_window: list or None, [start, end] in seconds for baseline subtraction
        contrast_conditions: list or None, keys to plot from dffTrace_stimuli
        contrast_labels: list or None, labels for the legend
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    # Defaults for rewarded trials of all contrasts
    if contrast_conditions is None:
        contrast_conditions = ['0.5 Rewarded', '0.25 Rewarded', '0.125 Rewarded', '0.0625 Rewarded', '0 Rewarded']
    if contrast_labels is None:
        contrast_labels = ['0.5 Rewarded', '0.25 Rewarded', '0.125 Rewarded', '0.0625 Rewarded', '0 Rewarded']
    colors = sns.color_palette("viridis", len(contrast_conditions))
    first_condition = contrast_conditions[0]
    if dffTrace_stimuli.get(first_condition) is None:
        print(f"No data available for condition {first_condition}")
        return
    n_cells = dffTrace_stimuli[first_condition].shape[0]
    n_rows = int(np.ceil(n_cells / n_cols))
    fps = 30
    pre_frames = int(abs(time_window[0]) * fps)
    post_frames = int(time_window[1] * fps)
    total_frames = pre_frames + post_frames
    time_axis = np.linspace(time_window[0], time_window[1], total_frames)
    if baseline_window is not None:
        baseline_start_idx = int((baseline_window[0] - time_window[0]) * fps)
        baseline_end_idx = int((baseline_window[1] - time_window[0]) * fps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cells == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    for cell_idx in range(n_cells):
        ax = axes[cell_idx]
        for cond_idx, condition in enumerate(contrast_conditions):
            if dffTrace_stimuli.get(condition) is not None:
                if cell_idx >= dffTrace_stimuli[condition].shape[0]:
                    continue
                cell_data = dffTrace_stimuli[condition][cell_idx, :, :]
                if baseline_window is not None:
                    baseline_vals = np.nanmean(cell_data[baseline_start_idx:baseline_end_idx, :], axis=0, keepdims=True)
                    cell_data = cell_data - baseline_vals
                mean_trace = np.nanmean(cell_data, axis=1)
                sem_trace = np.nanstd(cell_data, axis=1) / np.sqrt(np.sum(~np.isnan(cell_data), axis=1))
                ax.plot(time_axis, mean_trace, color=colors[cond_idx], 
                        linewidth=2, label=f'{contrast_labels[cond_idx]}')
                ax.fill_between(time_axis, 
                                mean_trace - sem_trace, 
                                mean_trace + sem_trace, 
                                color=colors[cond_idx], alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.set_title(f'Cell {cell_idx + 1}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_window)
    for idx in range(n_cells, len(axes)):
        fig.delaxes(axes[idx])
    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10, title='Contrast')
    fig.suptitle(f'Individual Cell Traces by Contrast - {session_name}', fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'{session_name}_individual_cell_traces_by_contrast.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Individual cell traces saved to: {save_file}")
    else:
        plt.show()
    return fig

def plot_individual_cell_traces_by_contrast_zscored(dffTrace_stimuli_z, session_name, save_path=None, 
                                                    figsize=(7.5, 30), n_cols=3, time_window=[-2, 6], baseline_window=None,
                                                    contrast_conditions=None, contrast_labels=None):
    """
    Plots individual cell traces (z-scored) aligned to stimulus, separated by contrast with viridis colormap.
    Allows baseline subtraction.
    Allows specifying contrast conditions and labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    if contrast_conditions is None:
        contrast_conditions = ['0.5 reward', '0.25 reward', '0.125 reward', '0.0625 reward', '0 reward']
    if contrast_labels is None:
        contrast_labels = ['0.5', '0.25', '0.125', '0.0625', '0']
    colors = sns.color_palette("viridis", len(contrast_conditions))
    first_condition = contrast_conditions[0]
    if dffTrace_stimuli_z.get(first_condition) is None:
        print(f"No data available for condition {first_condition}")
        return
    n_cells = dffTrace_stimuli_z[first_condition].shape[0]
    n_rows = int(np.ceil(n_cells / n_cols))
    fps = 30
    pre_frames = int(abs(time_window[0]) * fps)
    post_frames = int(time_window[1] * fps)
    total_frames = pre_frames + post_frames
    time_axis = np.linspace(time_window[0], time_window[1], total_frames)
    if baseline_window is not None:
        baseline_start_idx = int((baseline_window[0] - time_window[0]) * fps)
        baseline_end_idx = int((baseline_window[1] - time_window[0]) * fps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cells == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    for cell_idx in range(n_cells):
        ax = axes[cell_idx]
        for cond_idx, condition in enumerate(contrast_conditions):
            if dffTrace_stimuli_z.get(condition) is not None:
                if cell_idx >= dffTrace_stimuli_z[condition].shape[0]:
                    continue
                cell_data = dffTrace_stimuli_z[condition][cell_idx, :, :]
                if baseline_window is not None:
                    baseline_vals = np.nanmean(cell_data[baseline_start_idx:baseline_end_idx, :], axis=0, keepdims=True)
                    cell_data = cell_data - baseline_vals
                mean_trace = np.nanmean(cell_data, axis=1)
                sem_trace = np.nanstd(cell_data, axis=1) / np.sqrt(np.sum(~np.isnan(cell_data), axis=1))
                ax.plot(time_axis, mean_trace, color=colors[cond_idx], 
                        linewidth=2, label=f'{contrast_labels[cond_idx]}')
                ax.fill_between(time_axis, 
                                mean_trace - sem_trace, 
                                mean_trace + sem_trace, 
                                color=colors[cond_idx], alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F (z-score)')
        ax.set_title(f'Cell {cell_idx + 1}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_window)
    for idx in range(n_cells, len(axes)):
        fig.delaxes(axes[idx])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10, title='Contraste')
    fig.suptitle(f'Individual Cell Traces by Contrast (Z-scored) - {session_name}', fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'{session_name}_individual_cell_traces_by_contrast_zscored.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Individual cell traces (z-scored) saved to: {save_file}")
    else:
        plt.show()
    return fig

def plot_single_neuron_traces_by_contrast(dffTrace_stimuli, session_name, save_path=None, 
                                          time_window=[-2, 6], baseline_window=None,
                                          contrast_conditions=None, contrast_labels=None, fps=30):
    """
    Plot the traces of single neurons for each contrast condition.
    If session_name ends with 'MBL014', use contrast_conditions=['0.5 Rewarded', '0 Rewarded'] and contrast_labels=['0.5 Rewarded', '0 Rewarded'] by default.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    
    if session_name.endswith('MBL014'):
        if contrast_conditions is None:
            contrast_conditions = ['0.5 Rewarded', '0 Rewarded']
        if contrast_labels is None:
            contrast_labels = ['0.5 Rewarded', '0 Rewarded']

    if contrast_conditions is None:
        contrast_conditions = ['0.5 Rewarded', '0.25 Rewarded', '0.125 Rewarded', '0.0625 Rewarded', '0 Rewarded']
    if contrast_labels is None:
        contrast_labels = ['0.5 Rewarded', '0.25 Rewarded', '0.125 Rewarded', '0.0625 Rewarded', '0 Rewarded']
    colors = sns.color_palette("viridis", len(contrast_conditions))
    first_condition = contrast_conditions[0]
    if dffTrace_stimuli.get(first_condition) is None:
        print(f"No data available for condition {first_condition}")
        return
    n_cells = dffTrace_stimuli[first_condition].shape[0]
    pre_frames = int(abs(time_window[0]) * fps)
    post_frames = int(time_window[1] * fps)
    total_frames = pre_frames + post_frames
    time_axis = np.linspace(time_window[0], time_window[1], total_frames)
    if baseline_window is not None:
        baseline_start_idx = int((baseline_window[0] - time_window[0]) * fps)
        baseline_end_idx = int((baseline_window[1] - time_window[0]) * fps)
    # Create output folder
    if save_path is not None:
        single_neuron_dir = os.path.join(save_path, 'single-neuron plots')
        os.makedirs(single_neuron_dir, exist_ok=True)
    else:
        single_neuron_dir = None
    # Loop over neurons
    for cell_idx in range(n_cells):
        plt.figure(figsize=(8, 4))
        for cond_idx, condition in enumerate(contrast_conditions):
            if dffTrace_stimuli.get(condition) is not None and cell_idx < dffTrace_stimuli[condition].shape[0]:
                cell_data = dffTrace_stimuli[condition][cell_idx, :, :]
                n_trials = cell_data.shape[1]
                if n_trials == 0:
                    continue
                if baseline_window is not None:
                    baseline_vals = np.nanmean(cell_data[baseline_start_idx:baseline_end_idx, :], axis=0, keepdims=True)
                    cell_data = cell_data - baseline_vals
                mean_trace = np.nanmean(cell_data, axis=1)
                sem_trace = np.nanstd(cell_data, axis=1) / np.sqrt(np.sum(~np.isnan(cell_data), axis=1))
                plt.plot(time_axis, mean_trace, color=colors[cond_idx], linewidth=2, label=f'{contrast_labels[cond_idx]} (n={n_trials})')
                plt.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=colors[cond_idx], alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel('Time from stimulus onset (s)', fontsize=10)
        plt.ylabel('dF/F', fontsize=10)
        plt.title(f'{session_name} - Cell {cell_idx+1}', fontsize=11)
        plt.xlim(time_window)
        plt.legend(title='Contrast', fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
        plt.tight_layout(rect=[0, 0, 0.8, 1]) 
        plt.grid(True, alpha=0.3)
        
        
        # Save figure
        if single_neuron_dir is not None:
            fname = os.path.join(single_neuron_dir, f'{session_name}_cell{cell_idx+1}.png')
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    print("Done plotting all single-neuron traces.")

def plot_mean_dff_by_contrast(recordingList, event_type='stimulus', 
                             time_window=[0.1, 0.8], save_path=None, title=None, 
                             subfolder='responsive_neurons', dffTrace_mean_dict=None, use_zscored=True, baseline_window=[-0.2, 0],
                             contrasts_rewarded=None, contrast_values=None):
    """
    Plot mean df/f for a specific time window after an event as a function of contrast.
    Uses only rewarded trials for different contrasts.
    Connects the points from the same session with a dashed line.
    Shows the global mean as a black point with SEM.
    Performs baseline subtraction using the baseline_window before calculating the mean in the time window.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os
    
    # Define frame rate and time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before stimulus/event
    total_time = 8  # seconds (for time axis)
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
    
    # Convert time window from seconds to frame indices
    start_frame = int((time_window[0] + pre_stim_sec) * fRate_imaging)
    end_frame = int((time_window[1] + pre_stim_sec) * fRate_imaging)
    
    # Definir contrasts_rewarded y contrast_values si no se pasan como argumento
    if contrasts_rewarded is None:
        contrasts_rewarded = ['0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded']
    if contrast_values is None:
        contrast_values = [0.0, 0.0625, 0.125, 0.25, 0.5]  # Numeric values for plotting
    contrast_colors = sns.color_palette('viridis', len(contrasts_rewarded))[::-1] 
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Dictionary to store data for each contrast
    contrast_data = {contrast: [] for contrast in contrasts_rewarded}
    # To connect the points of each session
    session_points = {}
    
    # Process each session
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            
            # Load the data for this session
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            
            try:
                # Load mean df/f data
                if use_zscored:
                    pkl_name = 'imaging-dffTrace_mean_zscored.pkl'
                else:
                    pkl_name = 'imaging-dffTrace_mean.pkl'
                with open(os.path.join(subfolder_path, pkl_name), 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                
                # Select the appropriate dictionary based on event_type
                if event_type.lower() == 'stimulus':
                    dffTrace_mean = dffTrace_mean_stimuli
                elif event_type.lower() == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                elif event_type.lower() == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                else:
                    raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")
                
                # For this session, save the points to connect them later
                session_contrasts = []
                session_means = []
                
                # Calculate mean df/f for each rewarded contrast in the specified time window, after baseline subtraction
                for i, contrast in enumerate(contrasts_rewarded):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        # Ensure we have enough frames
                        if dffTrace_mean[contrast].shape[1] > end_frame:
                            # Baseline subtraction
                            mean_trace = np.nanmean(dffTrace_mean[contrast], axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            # Subtract baseline from all cells' traces
                            dff_bs = dffTrace_mean[contrast] - baseline
                            # Calculate mean across cells and time window
                            mean_dff = np.nanmean(dff_bs[:, start_frame:end_frame])
                            contrast_data[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values[i]  # Use numeric value for plotting
                            })
                            session_contrasts.append(contrast_values[i])
                            session_means.append(mean_dff)
                # Save the session points if there are at least two
                if len(session_contrasts) > 1:
                    session_points[session_name] = (session_contrasts, session_means)
                
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue
    
    # Connect the points of each session with a dashed line
    for session_name, (xvals, yvals) in session_points.items():
        ax.plot(xvals, yvals, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
    
    # Plot individual points per contrast
    for i, contrast in enumerate(contrasts_rewarded):
        if contrast_data[contrast]:
            df_contrast = pd.DataFrame(contrast_data[contrast])
            ax.scatter(df_contrast['contrast'], df_contrast['mean_dff'], 
                      color=contrast_colors[i], alpha=0.7, s=50, 
                      label=f'Contrast {contrast_values[i]} (Rewarded)', zorder=2)
    
    # Calculate and plot the global mean and SEM for each contrast
    # Agrupar por valor numérico de contraste
    grouped_data = defaultdict(list)
    for i, contrast in enumerate(contrasts_rewarded):
        val = contrast_values[i]
        for d in contrast_data[contrast]:
            grouped_data[val].append(d['mean_dff'])

    means = []
    sems = []
    x_vals = []
    for val in sorted(grouped_data.keys()):
        vals = grouped_data[val]
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals)/np.sqrt(len(vals)))
            x_vals.append(val)
    # Plot the global mean as a black line with SEM bars
    if x_vals:
        ax.plot(x_vals, means, color='k', linewidth=2, label='Mean across sessions', zorder=5)
        ax.errorbar(x_vals, means, yerr=sems, fmt='o', color='k', elinewidth=2, capsize=5, alpha=0.8, markersize=6, zorder=6)
    
    # Customize
    ax.set_xlabel('Contrast')
    ax.set_ylabel(f'Mean df/f')
    if title is None:
        title = f'Mean df/f by contrast ({time_window[0]}-{time_window[1]}s post-{event_type}) - Rewarded trials - {subfolder} (baseline subtracted)'
    ax.set_title(title, y=1.04, fontsize=14)
    ax.set_xscale('linear')
    ax.set_xticks(contrast_values)
    ax.set_xticklabels([str(c) for c in contrast_values])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'mean_dff_by_contrast_rewarded_{event_type}_{time_window[0]}_{time_window[1]}s_{subfolder}_baseline', save_path)
    plt.close(fig)
    return fig

def plot_peak_dff_by_contrast(recordingList, event_type='stimulus', 
                             time_window=[0.1, 0.8], save_path=None, title=None, 
                             subfolder='responsive_neurons', dffTrace_mean_dict=None, use_zscored=True):
    """
    Plot peak (max) df/f for a specific time window after an event as a function of contrast.
    Uses only rewarded trials for different contrasts.
    Join the points of the same session with a dotted line.
    Show the global mean as a short black line with SEM.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os
    
    # Define frame rate and time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before stimulus/event
    
    # Convert time window from seconds to frame indices
    start_frame = int((time_window[0] + pre_stim_sec) * fRate_imaging)
    end_frame = int((time_window[1] + pre_stim_sec) * fRate_imaging)
    
    # Define contrasts for rewarded trials and their corresponding numeric values
    contrasts_rewarded = ['0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded']
    contrast_values = [0.0, 0.0625, 0.125, 0.25, 0.5]  # Numeric values for plotting
    contrast_colors = sns.color_palette('viridis', len(contrasts_rewarded))[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Dictionary to store data for each contrast
    contrast_data = {contrast: [] for contrast in contrasts_rewarded}
    # To join the points of each session
    session_points = {}
    
    # Process each session
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            
            # Load the data for this session
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            
            try:
                # Load mean df/f data
                if use_zscored:
                    pkl_name = 'imaging-dffTrace_mean_zscored.pkl'
                else:
                    pkl_name = 'imaging-dffTrace_mean.pkl'
                with open(os.path.join(subfolder_path, pkl_name), 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                
                # Select the appropriate dictionary based on event_type
                if event_type.lower() == 'stimulus':
                    dffTrace_mean = dffTrace_mean_stimuli
                elif event_type.lower() == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                elif event_type.lower() == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                else:
                    raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")
                
                # For this session, save the points to join them later
                session_contrasts = []
                session_peaks = []
                
                # Calculate peak df/f for each rewarded contrast in the specified time window
                for i, contrast in enumerate(contrasts_rewarded):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        # Ensure we have enough frames
                        if dffTrace_mean[contrast].shape[1] > end_frame:
                            # Calculate peak across cells and time window
                            peak_dff = np.nanmax(dffTrace_mean[contrast][:, start_frame:end_frame])
                            contrast_data[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'peak_dff': peak_dff,
                                'contrast': contrast_values[i]  # Use numeric value for plotting
                            })
                            session_contrasts.append(contrast_values[i])
                            session_peaks.append(peak_dff)
                # Save the points of the session if there are at least two
                if len(session_contrasts) > 1:
                    session_points[session_name] = (session_contrasts, session_peaks)
                
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue
    
    # Join the points of each session with a dotted line
    for session_name, (xvals, yvals) in session_points.items():
        ax.plot(xvals, yvals, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
    
    # Plot individual points by contrast
    for i, contrast in enumerate(contrasts_rewarded):
        if contrast_data[contrast]:
            df_contrast = pd.DataFrame(contrast_data[contrast])
            ax.scatter(df_contrast['contrast'], df_contrast['peak_dff'], 
                      color=contrast_colors[i], alpha=0.7, s=50, 
                      label=f'Contrast {contrast_values[i]} (Rewarded)', zorder=2)
    
    # Calculate and plot the global mean and SEM for each contrast
    means = []
    sems = []
    for i, contrast in enumerate(contrasts_rewarded):
        vals = [d['peak_dff'] for d in contrast_data[contrast]]
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    # Plot the global mean as a short horizontal line (with SEM)
    for i, (x, m, s) in enumerate(zip(contrast_values, means, sems)):
        if not np.isnan(m):
            ax.errorbar(x, m, yerr=s, fmt='_', color='k', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    
    # Customize
    ax.set_xlabel('Contrast')
    ax.set_ylabel(f'Peak df/f')
    if title is None:
        title = f'Peak df/f by contrast ({time_window[0]}-{time_window[1]}s post-{event_type}) - Rewarded trials - {subfolder}'
    ax.set_title(title)
    ax.set_xscale('linear')
    ax.set_xticks(contrast_values)
    ax.set_xticklabels([str(c) for c in contrast_values])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'peak_dff_by_contrast_rewarded_{event_type}_{time_window[0]}_{time_window[1]}s_{subfolder}', save_path)
    plt.close(fig)
    return fig

def plot_cellwise_contra_ipsi_diff_by_ypix(recordingList, subfolder='all_neurons', recside_json_path=None, save_path=None):
    """
    For each cell from all sessions, plot the mean difference in z-scored df/f between
    'Choice Hemi Contra' and 'Choice Hemi Ipsi' as a function of the ypix coordinate.
    The recording side is read from a JSON file (recside_json_path, global for all sessions).
    If the recording side is 'left', ypix is set to 500 - ypix.
    If subfolder is 'responsive_neurons', only responsive neurons are used (index recalculated as in pipeline).
    Each session is shown in a different color. A global trend line is added.
    If save_path is specified, saves the figure in that folder; otherwise, shows it.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pickle
    import os
    import json
    from scipy.stats import linregress
    import utils_funcs as utils 

    if recside_json_path is None:
        raise ValueError('You must provide recside_json_path (path to recside_data.json)')
    with open(recside_json_path, 'r') as f:
        recside_dict = json.load(f)

    all_ypix = []
    all_diff = []
    all_session = []
    session_colors = sns.color_palette('tab10', n_colors=len(recordingList))

   
    pre_frames = 60  # 2 sec * 30 Hz
    post_frames = 180  # 6 sec * 30 Hz

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            rec_side = recside_dict.get(session_name, '').strip().lower()
            try:
                # Load zscored data
                with open(os.path.join(subfolder_path, 'imaging-dffTrace_mean_zscored.pkl'), 'rb') as f:
                    _, _, dffTrace_mean_choice = pickle.load(f)
                # Load stat
                imData = pd.read_pickle(pathname + 'imaging-data.pkl')
                stat = imData['stat']
                # Get ypix for each cell
                ypix = np.array([np.mean(cell['ypix']) for cell in stat])
                # Flip ypix if recording side is left (500 - ypix)
                if rec_side == 'left':
                    ypix = 500 - ypix
                # Get means for each cell, matching number of cells
                contra = dffTrace_mean_choice.get('Choice Hemi Contra')
                ipsi = dffTrace_mean_choice.get('Choice Hemi Ipsi')
                if contra is not None and ipsi is not None:
                    n_cells = min(len(ypix), contra.shape[0], ipsi.shape[0])
                    ypix = ypix[:n_cells]
                    # Baseline subtraction: baseline window -1 a 0 s 
                    fRate_imaging = 30
                    pre_stim_sec = 2
                    baseline_start_frame = int((-1 + pre_stim_sec) * fRate_imaging)  # 30
                    baseline_end_frame = int((0 + pre_stim_sec) * fRate_imaging)     # 60
                    # Subtract baseline from each trace
                    contra_bl = contra[:n_cells, :] - np.nanmean(contra[:n_cells, baseline_start_frame:baseline_end_frame], axis=1, keepdims=True)
                    ipsi_bl   = ipsi[:n_cells, :]   - np.nanmean(ipsi[:n_cells, baseline_start_frame:baseline_end_frame], axis=1, keepdims=True)
                    mean_contra = np.nanmean(contra_bl, axis=1)
                    mean_ipsi = np.nanmean(ipsi_bl, axis=1)
                    diff = mean_contra - mean_ipsi
                    # If responsive_neurons, recalculate the index and filter
                    if subfolder == 'responsive_neurons':
                        responsive_idx_dict = utils.filter_responsive_neurons(dffTrace_mean_choice, pre_frames, post_frames)
                        # Join all responsive indices
                        responsive_any = None
                        for cond, mask in responsive_idx_dict.items():
                            if mask is not None:
                                if responsive_any is None:
                                    responsive_any = mask.copy()
                                else:
                                    responsive_any = responsive_any | mask
                        if responsive_any is not None:
                            ypix = ypix[responsive_any]
                            diff = diff[responsive_any]
                    all_ypix.extend(ypix)
                    all_diff.extend(diff)
                    all_session.extend([session_name]*len(diff))
            except Exception as e:
                print(f"Error in session {session_name}: {str(e)}")
                continue
    # Convert to arrays
    all_ypix = np.array(all_ypix)
    all_diff = np.array(all_diff)
    all_session = np.array(all_session)
    # Plot
    plt.figure(figsize=(6, 3))
    unique_sessions = np.unique(all_session)
    for i, sess in enumerate(unique_sessions):
        mask = all_session == sess
        plt.scatter(all_ypix[mask], all_diff[mask], label=sess, alpha=0.7, s=20, color=session_colors[i % len(session_colors)])
    # Global trend line
    if len(all_ypix) > 1:
        slope, intercept, r, p, stderr = linregress(all_ypix, all_diff)
        xfit = np.linspace(np.min(all_ypix), np.max(all_ypix), 100)
        yfit = slope * xfit + intercept
        plt.plot(xfit, yfit, 'k--', linewidth=2, label=f'Global trend (r={r:.2f})')
    plt.xlabel('y (pix, corrected for recording side)')
    plt.ylabel('Δ df/f zscored (Contra - Ipsi)')
    plt.title('Mean Contra - Ipsi difference by cell location')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        save_figureAll('ypix_hemi_choiceDiff', save_path)
    plt.close()

def plot_reward_aligned_stim_ipsi_contra_diff(session_path, session_name, subfolder='responsive_neurons', save_path=None):
    """
    Plot the difference (rewarded - unrewarded) of neural activity aligned to reward for:
    - 'Stim Hemi Ipsi'
    - 'Stim Hemi Contra'
    Only these two conditions are plotted, each as a single curve (difference), in the same plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os

    # Load mean reward-aligned traces
    subfolder_path = os.path.join(session_path, subfolder)
    with open(os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl'), 'rb') as f:
        dffTrace_mean_reward, _, _ = pickle.load(f)

    # Define conditions and colors
    stim_conditions = [('Stim Hemi Ipsi', 'blue'), ('Stim Hemi Contra', 'red')]
    reward_types = [('Rewarded', '-'), ('Unrewarded', '--')]

    # Time axis (-2 to 6 s, 30 Hz)
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    plt.figure(figsize=(7, 4))
    for condition, color in stim_conditions:
        # Get rewarded and unrewarded traces
        trace_rewarded = dffTrace_mean_reward.get(condition)
        trace_unrewarded = dffTrace_mean_reward.get('Unrewarded')
        if trace_rewarded is not None and trace_unrewarded is not None:
            mean_rewarded = np.nanmean(trace_rewarded, axis=0)
            mean_unrewarded = np.nanmean(trace_unrewarded, axis=0)
            diff_trace = mean_rewarded - mean_unrewarded
            # SEM for the difference
            sem_rewarded = np.nanstd(trace_rewarded, axis=0) / np.sqrt(trace_rewarded.shape[0])
            sem_unrewarded = np.nanstd(trace_unrewarded, axis=0) / np.sqrt(trace_unrewarded.shape[0])
            sem_diff = np.sqrt(sem_rewarded**2 + sem_unrewarded**2)
            plt.plot(time_axis[:len(diff_trace)], diff_trace, color=color, label=condition + ' (Rewarded - Unrewarded)', linewidth=2)
            plt.fill_between(
                time_axis[:len(diff_trace)],
                diff_trace - sem_diff,
                diff_trace + sem_diff,
                color=color, alpha=0.3
            )
    plt.axvline(0, color='k', linestyle=':', linewidth=1)
    plt.xlabel('Time from reward (s)')
    plt.ylabel('df/f (Rewarded - Unrewarded)')
    plt.title(f'{session_name} - Reward-aligned: Stim Hemi Ipsi/Contra')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'{session_name}_reward_aligned_stim_hemi_ipsi_contra_diff', save_path)
    plt.close()

def plot_aligned_by_stim_side_contrasts(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions_left=None,
    contrast_conditions_right=None,
    alignment='reward',  # 'reward', 'choice' or 'stimulus'
    baseline_window=[-0.2, 0]  # in seconds
):
    """
    Plot neural activity aligned to reward, choice, or stimulus for different contrasts, separated by stimulus side (Left vs Right).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import seaborn as sns

    print(f"Starting {alignment}-aligned activity by stim side and contrasts for all sessions...")

    # Default contrast conditions
    if contrast_conditions_left is None:
        if alignment == 'reward':
            contrast_conditions_left = [
                '-0.0625 Rewarded', '-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded'
            ]
        elif alignment in ['stimulus', 'stim']:
            contrast_conditions_left = [
                '-0.0625 Rewarded', '-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded'
            ]
        else:  # choice
            contrast_conditions_left = [
                 '-0.0625 Rewarded', '-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded'
            ]
    if contrast_conditions_right is None:
        if alignment == 'reward':
            contrast_conditions_right = [
                '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
            ]
        elif alignment in ['stimulus', 'stim']:
            contrast_conditions_right = [
                '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
            ]
        else:  # choice
            contrast_conditions_right = [
                 '-0.0625 Rewarded', '-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded'
            ]
    colormap = 'viridis'
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            session_name = recordingList.sessionName.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            print(f'Plotting {alignment}-aligned activity by stim side for session: {session_name}')
            try:
                # adjust contrast_conditions according to the session_name
                if contrast_conditions_left is None or contrast_conditions_right is None:
                    if session_name.endswith('MBL014'):
                        contrast_conditions_left = ['-0.5 Rewarded']
                        contrast_conditions_right = ['0.5 Rewarded']
                    else:
                        contrast_conditions_left = ['-0.5 Rewarded', '-0.25 Rewarded', '-0.125 Rewarded', '-0.0625 Rewarded']
                        contrast_conditions_right = ['0.5 Rewarded', '0.25 Rewarded', '0.125 Rewarded', '0.0625 Rewarded']
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                # Select the correct dictionary
                if alignment == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment in ['stimulus', 'stim']:
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    print(f"Unknown alignment: {alignment}")
                    continue
                # Time axis
                fRate = 30
                pre_stim_sec = 2
                total_time = 8
                n_frames = int(total_time * fRate)
                time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                def plot_stim_side(ax, contrast_conditions, panel_title):
                    colors = sns.color_palette(colormap, len(contrast_conditions))[::-1] 
                    for i, contrast_cond in enumerate(contrast_conditions):
                        data = dffTrace_mean.get(contrast_cond)
                        if data is not None and np.size(data) > 0:
                            mean_trace = np.nanmean(data, axis=0)
                            # Baseline subtraction
                            fRate = 30
                            pre_stim_sec = 2
                            total_time = 8
                            n_frames = int(total_time * fRate)
                            time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
                            if baseline_window is not None:
                                # Find indices corresponding to baseline_window
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1  # asegurar al menos un frame
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                # Default: last 200 ms before the event
                                baseline_frames = int(0.2 * fRate)
                                baseline = np.nanmean(mean_trace[fRate*2-baseline_frames:fRate*2])  # pre_stim_sec=2s
                            mean_trace_bs = mean_trace - baseline
                            sem_trace = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
                            sem_trace_bs = sem_trace  # SEM no cambia con baseline
                            ax.plot(time_axis[:len(mean_trace_bs)], mean_trace_bs,
                                    color=colors[i], label=contrast_cond, linewidth=2)
                            ax.fill_between(
                                time_axis[:len(mean_trace_bs)],
                                mean_trace_bs - sem_trace_bs,
                                mean_trace_bs + sem_trace_bs,
                                color=colors[i], alpha=0.3
                            )
                    ax.axvline(0, color='k', linestyle=':', linewidth=1)
                    ax.set_xlabel(f'Time from {alignment.capitalize()} (s)')
                    ax.set_ylabel('df/f')
                    ax.set_title(panel_title)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='both', labelsize=18)

                plot_stim_side(ax1, contrast_conditions_left, 'Left Stimuli')
                plot_stim_side(ax2, contrast_conditions_right, 'Right Stimuli')
                title = f'{session_name} - Calcium activity by stimulus side and contrast'
                fig.suptitle(title, y=1.02)
                plt.tight_layout()
                
                if save_path is not None:
                    save_figureAll(f'{session_name}_{alignment}_aligned_by_stim_side_contrasts.png', save_path)
                plt.close()
                print(f"✓ Successfully plotted and saved for session: {session_name}")
            except Exception as e:
                print(f"✗ Error plotting session {session_name}: {str(e)}")
                continue
    print(f"Completed {alignment}-aligned activity by stim side and contrasts for all sessions.")

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
    plt.figure(figsize=(9, 2.5))
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

def plot_event_aligned_by_bias_type(session_path, session_name, save_path=None, bias_type='Choice Bias', alignment='choice'):
    """
    Plot the average activity of responsive neurons aligned to the specified event (reward, stimulus, or choice),
    separated by bias_type ('Choice Bias Ipsi' vs 'Choice Bias Contra' o 'Stim Bias Ipsi' vs 'Stim Bias Contra').
    The argument bias_type must be 'Choice Bias' or 'Stim Bias'.
    The argument alignment must be 'reward', 'stimulus' or 'choice'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    
    # Load activity data
    pkl_file = os.path.join(session_path, 'responsive_neurons', 'imaging-dffTrace_mean.pkl')
    if not os.path.exists(pkl_file):
        print(f"File not found: {pkl_file}")
        return

    with open(pkl_file, 'rb') as f:
        dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)

    # Select the dictionary according to alignment
    if alignment.lower() == 'reward':
        dffTrace_mean = dffTrace_mean_reward
    elif alignment.lower() == 'stimulus':
        dffTrace_mean = dffTrace_mean_stimuli
    elif alignment.lower() == 'choice':
        dffTrace_mean = dffTrace_mean_choice
    else:
        print("alignment must be 'reward', 'stimulus' or 'choice'")
        return

    # Determine the key pair according to bias_type
    if bias_type.lower().startswith('choice'):
        key_ipsi = 'Choice Bias Ipsi'
        key_contra = 'Choice Bias Contra'
    elif bias_type.lower().startswith('stim'):
        key_ipsi = 'Stim Bias Ipsi'
        key_contra = 'Stim Bias Contra'
    else:
        print("bias_type must be 'Choice Bias' or 'Stim Bias'")
        return

    data_ipsi = dffTrace_mean.get(key_ipsi)
    data_contra = dffTrace_mean.get(key_contra)

    if data_ipsi is None or data_contra is None:
        print(f"No data for '{key_ipsi}' or '{key_contra}' in alignment '{alignment}'")
        return

    # Compute mean and SEM
    mean_ipsi = np.nanmean(data_ipsi, axis=0)
    sem_ipsi = np.nanstd(data_ipsi, axis=0) / np.sqrt(data_ipsi.shape[0])
    mean_contra = np.nanmean(data_contra, axis=0)
    sem_contra = np.nanstd(data_contra, axis=0) / np.sqrt(data_contra.shape[0])

    # Time axis (assuming -2 to 6 s, 30 Hz)
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis[:len(mean_ipsi)], mean_ipsi, label='Ipsi', color='royalblue')
    plt.fill_between(time_axis[:len(mean_ipsi)], mean_ipsi - sem_ipsi, mean_ipsi + sem_ipsi, color='royalblue', alpha=0.3)
    plt.plot(time_axis[:len(mean_contra)], mean_contra, label='Contra', color='orange')
    plt.fill_between(time_axis[:len(mean_contra)], mean_contra - sem_contra, mean_contra + sem_contra, color='orange', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(f'Time from {alignment.lower()} (s)')
    plt.ylabel('dF/F')
    plt.title(f'{session_name} - Calcium activity by {bias_type}')
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_figureAll(f'{session_name}_{alignment}_aligned_by_{bias_type.replace(" ", "")}', save_path)
    plt.close()

def plot_AP_ML_hemi_contra_ipsi(df, title=None, save_path=None, alignment='choice'):
    """
    Plots AP (y) vs ML (x) for each cell, coloring by the sign of contra_ipsi_diff and sizing by its magnitude.
    - Red: contra > ipsi (positive diff)
    - Blue: ipsi > contra (negative diff)
    - Size: proportional to |contra_ipsi_diff|
    alignment: 'choice' or 'stimulus' (used in title and filename)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # Color: blue for negative, red for positive
    colors = np.where(df['contra_ipsi_diff'] >= 0, 'red', 'blue')
    # Size: proportional to magnitude (scaled for visibility)
    if np.nanmax(np.abs(df['contra_ipsi_diff'])) > 0:
        sizes = 30 + 120 * np.abs(df['contra_ipsi_diff']) / np.nanmax(np.abs(df['contra_ipsi_diff']))
    else:
        sizes = 30  # fallback if all diffs are zero

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df['ML'], df['AP'],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolor='k'
    )
    plt.xlabel('ML (mm)')
    plt.ylabel('AP (mm)')
    if title is None:
        title = f'AP vs ML (color=sign, size=magnitude) - {alignment.capitalize()} aligned'
    plt.title(title)
    # Add legend for color
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Contra > Ipsi', markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Ipsi > Contra', markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'AP_ML_hemi_{alignment}Diff', save_path)
    plt.close()

def plot_AP_ML_contra_ipsi(df, title='AP vs ML (color=sign, size=magnitude)', save_path=None):
    """
    Plots AP (y) vs ML (x) for each cell, coloring by the sign of contra_ipsi_diff and sizing by its magnitude.
    - Red: contra > ipsi (positive diff)
    - Blue: ipsi > contra (negative diff)
    - Size: proportional to |contra_ipsi_diff|
    If save_path is specified, saves the figure in that folder as 'AP_ML_choiceDiff.png'; otherwise, shows it.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Color: blue for negative, red for positive
    colors = np.where(df['contra_ipsi_diff'] >= 0, 'red', 'blue')
    # Size: proportional to magnitude (scaled for visibility)
    if np.nanmax(np.abs(df['contra_ipsi_diff'])) > 0:
        sizes = 30 + 120 * np.abs(df['contra_ipsi_diff']) / np.nanmax(np.abs(df['contra_ipsi_diff']))
    else:
        sizes = 30  # fallback if all diffs are zero

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df['ML'], df['AP'],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolor='k'
    )
    plt.xlabel('ML (mm)')
    plt.ylabel('AP (mm)')
    plt.title(title)
    # Add legend for color
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Contra > Ipsi', markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Ipsi > Contra', markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'AP_ML_choiceDiff.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"Figure saved at: {fname}")
        plt.close()
    else:
        plt.show()

def plot_mean_dff_by_contrast_hemi_ipsi_contra(
    recordingList,
    event_type='stimulus',
    time_window=[0.1, 0.8],
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions_ipsi=None,
    contrast_conditions_contra=None,
    use_zscored=True,
    title=None
):
    """
    Plot mean df/f for a specific time window after an event as a function of contrast, separated by ipsi/contra.
    ipsi and contra data are plotted in separate subplots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    # Time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before stimulus/event
    total_time = 8  # seconds (for time axis)
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Convert time window to frame indices
    start_frame = int((time_window[0] + pre_stim_sec) * fRate_imaging)
    end_frame = int((time_window[1] + pre_stim_sec) * fRate_imaging)

    # Default conditions if not provided
    if contrast_conditions_ipsi is None:
        contrast_conditions_ipsi = ['-0.0625 Rewarded Hemi Ipsi', '-0.125 Rewarded Hemi Ipsi', '-0.25 Rewarded Hemi Ipsi', '-0.5 Rewarded Hemi Ipsi', '0 Rewarded Hemi Ipsi', '0.0625 Rewarded Hemi Ipsi', '0.125 Rewarded Hemi Ipsi', '0.25 Rewarded Hemi Ipsi', '0.5 Rewarded Hemi Ipsi']
    if contrast_conditions_contra is None:
        contrast_conditions_contra = ['0.0625 Rewarded Hemi Contra', '0.125 Rewarded Hemi Contra', '0.25 Rewarded Hemi Contra', '0.5 Rewarded Hemi Contra', '0 Rewarded Hemi Contra', '0.0625 Rewarded Hemi Contra', '0.125 Rewarded Hemi Contra', '0.25 Rewarded Hemi Contra', '0.5 Rewarded Hemi Contra']
    contrast_values_ipsi = [-0.0625, -0.125, -0.25, -0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    contrast_values_contra = [-0.0625, -0.125, -0.25, -0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    colors_ipsi = sns.color_palette('Blues', len(contrast_conditions_ipsi))[::-1]
    colors_contra = sns.color_palette('Reds', len(contrast_conditions_contra))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    contrast_data_ipsi = {contrast: [] for contrast in contrast_conditions_ipsi}
    contrast_data_contra = {contrast: [] for contrast in contrast_conditions_contra}
    session_points_ipsi = {}
    session_points_contra = {}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                # IPSI
                session_contrasts_ipsi = []
                session_means_ipsi = []
                for i, contrast in enumerate(contrast_conditions_ipsi):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        if dffTrace_mean[contrast].shape[1] > end_frame:
                            mean_trace = np.nanmean(dffTrace_mean[contrast], axis=0)
                            baseline_frames = int(0.2 * fRate_imaging)
                            baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            dff_bs = dffTrace_mean[contrast] - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_frame:end_frame])
                            contrast_data_ipsi[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_ipsi[i]
                            })
                            session_contrasts_ipsi.append(contrast_values_ipsi[i])
                            session_means_ipsi.append(mean_dff)
                if len(session_contrasts_ipsi) > 1:
                    session_points_ipsi[session_name] = (session_contrasts_ipsi, session_means_ipsi)
                # CONTRA
                session_contrasts_contra = []
                session_means_contra = []
                for i, contrast in enumerate(contrast_conditions_contra):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        if dffTrace_mean[contrast].shape[1] > end_frame:
                            mean_trace = np.nanmean(dffTrace_mean[contrast], axis=0)
                            baseline_frames = int(0.2 * fRate_imaging)
                            baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            dff_bs = dffTrace_mean[contrast] - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_frame:end_frame])
                            contrast_data_contra[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_contra[i]
                            })
                            session_contrasts_contra.append(contrast_values_contra[i])
                            session_means_contra.append(mean_dff)
                if len(session_contrasts_contra) > 1:
                    session_points_contra[session_name] = (session_contrasts_contra, session_means_contra)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue
    # Plot points from each session with dashed line
    for session_name, (xvals, yvals) in session_points_ipsi.items():
        ax1.plot(xvals, yvals, linestyle='--', color='blue', alpha=0.5, linewidth=1, zorder=1)
    for session_name, (xvals, yvals) in session_points_contra.items():
        ax2.plot(xvals, yvals, linestyle='--', color='red', alpha=0.5, linewidth=1, zorder=1)
    # Plot individual points by contrast
    for i, contrast in enumerate(contrast_conditions_ipsi):
        if contrast_data_ipsi[contrast]:
            df_contrast = pd.DataFrame(contrast_data_ipsi[contrast])
            ax1.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                      color=colors_ipsi[i], alpha=0.7, s=50,
                      label=f'Ipsi {contrast_values_ipsi[i]}', zorder=2)
    for i, contrast in enumerate(contrast_conditions_contra):
        if contrast_data_contra[contrast]:
            df_contrast = pd.DataFrame(contrast_data_contra[contrast])
            ax2.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                      color=colors_contra[i], alpha=0.7, s=50,
                      label=f'Contra {contrast_values_contra[i]}', zorder=2)
    # Global mean and SEM
    means_ipsi = []
    sems_ipsi = []
    for i, contrast in enumerate(contrast_conditions_ipsi):
        vals = [d['mean_dff'] for d in contrast_data_ipsi[contrast]]
        if len(vals) > 0:
            means_ipsi.append(np.mean(vals))
            sems_ipsi.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_ipsi.append(np.nan)
            sems_ipsi.append(np.nan)
    means_contra = []
    sems_contra = []
    for i, contrast in enumerate(contrast_conditions_contra):
        vals = [d['mean_dff'] for d in contrast_data_contra[contrast]]
        if len(vals) > 0:
            means_contra.append(np.mean(vals))
            sems_contra.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_contra.append(np.nan)
            sems_contra.append(np.nan)
    for i, (x, m, s) in enumerate(zip(contrast_values_ipsi, means_ipsi, sems_ipsi)):
        if not np.isnan(m):
            ax1.errorbar(x, m, yerr=s, fmt='_', color='blue', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    for i, (x, m, s) in enumerate(zip(contrast_values_contra, means_contra, sems_contra)):
        if not np.isnan(m):
            ax2.errorbar(x, m, yerr=s, fmt='_', color='red', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    # Labels and titles
    ax1.set_xlabel('Contrast')
    ax1.set_ylabel(f'Mean df/f')
    ax1.set_title('Ipsi', fontsize=14)
    ax1.set_xscale('linear')
    ax1.set_xticks(contrast_values_ipsi)
    ax1.set_xticklabels([str(c) for c in contrast_values_ipsi])
    import matplotlib.pyplot as plt
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_xlabel('Contrast')
    ax2.set_ylabel(f'Mean df/f')
    ax2.set_title('Contra', fontsize=14)
    ax2.set_xscale('linear')
    ax2.set_xticks(contrast_values_contra)
    ax2.set_xticklabels([str(c) for c in contrast_values_contra])
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if title is not None:
        fig.suptitle(title, y=1.04, fontsize=16)
    else:
        fig.suptitle(f'Mean df/f by contrast ({time_window[0]}-{time_window[1]}s post-{event_type}) - Ipsi vs Contra - {subfolder} (baseline subtracted)', y=1.04, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        save_figureAll(f'mean_dff_by_contrast_hemi_ipsi_contra_{event_type}_{subfolder}_baseline', save_path)
    plt.close(fig)
    return fig

def plot_mean_dff_by_contrast_bias_ipsi_contra(
    recordingList,
    event_type='stimulus',
    time_window=[0.1, 0.8],
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions_bias=None,
    contrast_conditions_nobias=None,
    use_zscored=True,
    title=None
):
    """
    Plot the mean df/f for each contrast, collapsed into 5 absolute values (0, 0.0625, 0.125, 0.25, 0.5),
    combining positive and negative values into the same group, for bias ipsi and bias contra.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    # Define absolute contrast values
    abs_contrast_values = [0, 0.0625, 0.125, 0.25, 0.5]
    abs_contrast_labels = [str(c) for c in abs_contrast_values]
    contrast_colors = sns.color_palette('viridis', len(abs_contrast_values))[::-1]

    # Default conditions if not provided
    if contrast_conditions_bias is None:
        contrast_conditions_bias = [
            '-0.0625 Rewarded Bias Ipsi', '-0.125 Rewarded Bias Ipsi', '-0.25 Rewarded Bias Ipsi', '-0.5 Rewarded Bias Ipsi',
            '0 Rewarded Bias Ipsi', '0.0625 Rewarded Bias Ipsi', '0.125 Rewarded Bias Ipsi', '0.25 Rewarded Bias Ipsi', '0.5 Rewarded Bias Ipsi'
        ]
    if contrast_conditions_nobias is None:
        contrast_conditions_nobias = [
            '-0.0625 Rewarded Bias Contra', '-0.125 Rewarded Bias Contra', '-0.25 Rewarded Bias Contra', '-0.5 Rewarded Bias Contra',
            '0 Rewarded Bias Contra', '0.0625 Rewarded Bias Contra', '0.125 Rewarded Bias Contra', '0.25 Rewarded Bias Contra', '0.5 Rewarded Bias Contra'
        ]

    # Map each condition to its absolute contrast value
    def get_abs_contrast(cond):
        for val in abs_contrast_values:
            if cond.startswith(f'{val} ') or cond.startswith(f'-{val} '):
                return val
        return None

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    subplot_info = [
        (axes[0], contrast_conditions_bias, 'Bias Ipsi'),
        (axes[1], contrast_conditions_nobias, 'Bias Contra')
    ]

    # Store collapsed data by absolute value
    abs_contrast_data = {'Bias Ipsi': {v: [] for v in abs_contrast_values}, 'Bias Contra': {v: [] for v in abs_contrast_values}}
    session_points = {'Bias Ipsi': {}, 'Bias Contra': {}}

    # Process each session
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                # IPSI
                session_contrasts_ipsi = []
                session_means_ipsi = []
                for cond in contrast_conditions_bias:
                    abs_val = get_abs_contrast(cond)
                    if abs_val is not None and cond in dffTrace_mean and dffTrace_mean[cond] is not None:
                        mean_dff = np.nanmean(dffTrace_mean[cond])
                        abs_contrast_data['Bias Ipsi'][abs_val].append({'animal': animal_id, 'session': session_name, 'mean_dff': mean_dff, 'contrast': abs_val})
                        session_contrasts_ipsi.append(abs_val)
                        session_means_ipsi.append(mean_dff)
                if len(session_contrasts_ipsi) > 1:
                    session_points['Bias Ipsi'][session_name] = (session_contrasts_ipsi, session_means_ipsi)
                # CONTRA
                session_contrasts_contra = []
                session_means_contra = []
                for cond in contrast_conditions_nobias:
                    abs_val = get_abs_contrast(cond)
                    if abs_val is not None and cond in dffTrace_mean and dffTrace_mean[cond] is not None:
                        mean_dff = np.nanmean(dffTrace_mean[cond])
                        abs_contrast_data['Bias Contra'][abs_val].append({'animal': animal_id, 'session': session_name, 'mean_dff': mean_dff, 'contrast': abs_val})
                        session_contrasts_contra.append(abs_val)
                        session_means_contra.append(mean_dff)
                if len(session_contrasts_contra) > 1:
                    session_points['Bias Contra'][session_name] = (session_contrasts_contra, session_means_contra)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue

    # Plot
    for ax, _, group_name in subplot_info:
        # Connect points of each session
        for session_name, (xvals, yvals) in session_points[group_name].items():
            ax.plot(xvals, yvals, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
        # Individual points per absolute contrast
        for i, abs_val in enumerate(abs_contrast_values):
            if abs_contrast_data[group_name][abs_val]:
                df_contrast = pd.DataFrame(abs_contrast_data[group_name][abs_val])
                ax.scatter(df_contrast['contrast'], df_contrast['mean_dff'], 
                          color=contrast_colors[i], alpha=0.7, s=50, 
                          label=f'Contrast {abs_contrast_labels[i]}', zorder=2)
        # Global mean and SEM
        means = []
        sems = []
        for abs_val in abs_contrast_values:
            vals = [d['mean_dff'] for d in abs_contrast_data[group_name][abs_val]]
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(np.std(vals)/np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(np.nan)
        for i, (x, m, s) in enumerate(zip(abs_contrast_values, means, sems)):
            if not np.isnan(m):
                ax.errorbar(x, m, yerr=s, fmt='_', color='k', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
        ax.set_xlabel('Contrast (absolute value)')
        ax.set_title(f'Rewarded {group_name}')
        ax.set_xscale('linear')
        ax.set_xticks(abs_contrast_values)
        ax.set_xticklabels(abs_contrast_labels)
        import matplotlib.pyplot as plt
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_ylabel(f'Mean df/f')
    plt.tight_layout()
    if title is None:
        title = f'Mean df/f by absolute contrast ({time_window[0]}-{time_window[1]}s post-{event_type}) - Rewarded trials - {subfolder} (baseline subtracted)'
    fig.suptitle(title, y=1.04, fontsize=14)
    if save_path is not None:
        save_figureAll(f'mean_dff_by_contrast_rewarded_{event_type}_{time_window[0]}_{time_window[1]}s_{subfolder}_bias_ipsi_vs_contra_abs', save_path)
        plt.close(fig)
    else:
        plt.show()
    return fig

def plot_mean_sem_across_sessions_by_stim_side(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions_left=None,
    contrast_conditions_right=None,
    alignment='stimulus',  # 'reward', 'choice' or 'stimulus'
    baseline_window=[-0.2, 0]  # in seconds
):
    """
    Plot the mean and SEM across sessions for each contrast condition, separated by stimulus side (Left vs Right).
    Each subplot shows the mean trace and SEM across sessions for each contrast.
    Baseline subtraction is performed using the baseline_window (in seconds) if provided, otherwise the last 200 ms before the event is used by default.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import seaborn as sns

        # Default contrast conditions if not provided
    if contrast_conditions_left is None:
        contrast_conditions_left = [
            '-0.0625 Rewarded', '-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded'
        ]
    if contrast_conditions_right is None:
        contrast_conditions_right = [
            '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
        ]
    colormap = 'viridis'
    colors_left = sns.color_palette(colormap, len(contrast_conditions_left))
    colors_right = sns.color_palette(colormap, len(contrast_conditions_right))

    # Dictionaries to save the traces of each session by condition
    traces_left = {cond: [] for cond in contrast_conditions_left}
    traces_right = {cond: [] for cond in contrast_conditions_right}
    time_axis = None

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            session_name = recordingList.sessionName.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            try:
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                # Select the correct dictionary
                if alignment == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment in ['stimulus', 'stim']:
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    print(f"Unknown alignment: {alignment}")
                    continue
                # Time axis
                fRate = 30
                pre_stim_sec = 2
                total_time = 8
                n_frames = int(total_time * fRate)
                time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
                # Store traces for each condition
                for cond in contrast_conditions_left:
                    data = dffTrace_mean.get(cond)
                    if data is not None and np.size(data) > 0:
                        mean_trace = np.nanmean(data, axis=0)
                        # Baseline subtraction
                        if baseline_window is not None:
                            baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                            baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                            if baseline_end_idx == baseline_start_idx:
                                baseline_end_idx += 1
                            baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                        else:
                            baseline_frames = int(0.2 * fRate)
                            baseline = np.nanmean(mean_trace[fRate*2-baseline_frames:fRate*2])
                        mean_trace_bs = mean_trace - baseline
                        traces_left[cond].append(mean_trace_bs)
                for cond in contrast_conditions_right:
                    data = dffTrace_mean.get(cond)
                    if data is not None and np.size(data) > 0:
                        mean_trace = np.nanmean(data, axis=0)
                        # Baseline subtraction
                        if baseline_window is not None:
                            baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                            baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                            if baseline_end_idx == baseline_start_idx:
                                baseline_end_idx += 1
                            baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                        else:
                            baseline_frames = int(0.2 * fRate)
                            baseline = np.nanmean(mean_trace[fRate*2-baseline_frames:fRate*2])
                        mean_trace_bs = mean_trace - baseline
                        traces_right[cond].append(mean_trace_bs)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # Left
    for i, cond in enumerate(contrast_conditions_left):
        traces = np.array(traces_left[cond])
        if traces.shape[0] > 0:
            mean_across_sessions = np.nanmean(traces, axis=0)
            sem_across_sessions = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
            ax1.plot(time_axis[:len(mean_across_sessions)], mean_across_sessions, color=colors_left[i], label=cond, linewidth=2)
            ax1.fill_between(
                time_axis[:len(mean_across_sessions)],
                mean_across_sessions - sem_across_sessions,
                mean_across_sessions + sem_across_sessions,
                color=colors_left[i], alpha=0.3
            )
    ax1.axvline(0, color='k', linestyle=':', linewidth=1)
    ax1.set_xlabel(f"Time from '{alignment}' (s)")
    ax1.set_ylabel('df/f')
    ax1.set_title('Left Stimuli')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=18)
    # Right
    for i, cond in enumerate(contrast_conditions_right):
        traces = np.array(traces_right[cond])
        if traces.shape[0] > 0:
            mean_across_sessions = np.nanmean(traces, axis=0)
            sem_across_sessions = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
            ax2.plot(time_axis[:len(mean_across_sessions)], mean_across_sessions, color=colors_right[i], label=cond, linewidth=2)
            ax2.fill_between(
                time_axis[:len(mean_across_sessions)],
                mean_across_sessions - sem_across_sessions,
                mean_across_sessions + sem_across_sessions,
                color=colors_right[i], alpha=0.3
            )
    ax2.axvline(0, color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel(f"Time from '{alignment}' (s)")
    ax2.set_title('Right Stimuli')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'mean_sem_across_sessions_{alignment}_by_stim_side', save_path)
    plt.close(fig)
    return fig

def plot_peak_window_mean_dff_by_contrast(
    recordingList,
    event_type='stimulus',
    save_path=None,
    title=None,
    subfolder='responsive_neurons',
    dffTrace_mean_dict=None,
    use_zscored=True,
    baseline_window=[-0.2, 0],
    contrasts_rewarded=None,
    contrast_values=None
):
    """
    Plot mean df/f for a window of ±250 ms around the peak of the mean curve for each contrast and session.
    Connects the points from the same session with a dashed line.
    Shows the global mean as a black point with SEM.
    Performs baseline subtraction using the baseline_window before calculating the mean in the window.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    # Time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before the stimulus/event
    total_time = 8  # seconds (for the time axis)
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Define contrasts_rewarded and contrast_values if not passed as argument
    if contrasts_rewarded is None:
        contrasts_rewarded = ['0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded']
    if contrast_values is None:
        contrast_values = [0.0, 0.0625, 0.125, 0.25, 0.5]
    contrast_colors = sns.color_palette('viridis', len(contrasts_rewarded))[::-1]

    
    fig, ax = plt.subplots(figsize=(10, 4))

    # Dict to save data by contrast
    contrast_data = {contrast: [] for contrast in contrasts_rewarded}
    session_points = {}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                session_contrasts = []
                session_means = []
                for i, contrast in enumerate(contrasts_rewarded):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        data = dffTrace_mean[contrast]
                        if data.shape[1] > 0:
                            # Calculate the mean trace and find the peak
                            mean_trace = np.nanmean(data, axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            mean_trace_bs = mean_trace - baseline
                            # Find the peak in the post-stimulus window (0 to 2s)
                            peak_window = (time_axis >= 0) & (time_axis <= 2)
                            peak_idx = np.argmax(np.abs(mean_trace_bs[peak_window])) + np.where(peak_window)[0][0]
                            # Window of ±250 ms around the peak
                            window_size = int(0.25 * fRate_imaging)
                            start_idx = max(0, peak_idx - window_size)
                            end_idx = min(len(mean_trace_bs), peak_idx + window_size)
                            # Calculate the mean value in the window
                            dff_bs = data - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_idx:end_idx])
                            contrast_data[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values[i]
                            })
                            session_contrasts.append(contrast_values[i])
                            session_means.append(mean_dff)
                if len(session_contrasts) > 1:
                    session_points[session_name] = (session_contrasts, session_means)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue

    # Join the points of each session with a dotted line
    for session_name, (xvals, yvals) in session_points.items():
        ax.plot(xvals, yvals, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    # Plot individual points by contrast
    for i, contrast in enumerate(contrasts_rewarded):
        if contrast_data[contrast]:
            df_contrast = pd.DataFrame(contrast_data[contrast])
            ax.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                      color=contrast_colors[i], alpha=0.7, s=50,
                      label=f'Contrast {contrast_values[i]} (Rewarded)', zorder=2)

    # Calculate and plot the global mean and SEM for each contrast
    means = []
    sems = []
    for i, contrast in enumerate(contrasts_rewarded):
        vals = [d['mean_dff'] for d in contrast_data[contrast]]
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    # Black line connecting the mean points, only one label for the legend
    ax.plot(contrast_values, means, color='k', linewidth=2, zorder=3, label='Mean')
    for i, (x, m, s) in enumerate(zip(contrast_values, means, sems)):
        if not np.isnan(m):
            ax.errorbar(x, m, yerr=s, fmt='_', color='k', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    ax.set_xlabel('Contrast')
    ax.set_ylabel(f'Mean df/f (peak ±250ms)')
    if title is None:
        title = f'Mean df/f by contrast (±250 ms around peak, baseline subtracted) - Rewarded trials - {subfolder}'
    ax.set_title(title, y=1.04, fontsize=14)
    ax.set_xscale('linear')
    ax.set_xticks(contrast_values)
    ax.set_xticklabels([str(c) for c in contrast_values])
    import matplotlib.pyplot as plt
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    # Only one 'Mean' in the legend
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'peak_window_mean_dff_by_contrast_rewarded_{event_type}_{subfolder}_baseline', save_path)
    plt.close(fig)
    return fig

def plot_reward_aligned_stim_ipsi_contra_diff_across_sessions(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None,
    key_type='hemi'  # 'hemi' (default) or 'bias'
):
    """
    Plot the mean and SEM across sessions of the difference (rewarded - unrewarded) for:
    - 'Stim Hemi Ipsi' and 'Stim Hemi Contra' (key_type='hemi')
    - 'Stim Bias Ipsi' and 'Stim Bias Contra' (key_type='bias')
    Both curves are shown in a single plot, with SEM bands.
    The x-axis is labeled 'Time from reward (s)'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os

    if key_type == 'hemi':
        stim_conditions = [('Stim Hemi Ipsi', 'blue'), ('Stim Hemi Contra', 'red')]
        title_str = 'Reward-aligned: Stim Hemi Ipsi/Contra (mean ± SEM across sessions)'
        save_name = 'reward_aligned_stim_hemi_ipsi_contra_diff_across_sessions'
    elif key_type == 'bias':
        stim_conditions = [('Stim Bias Ipsi', 'blue'), ('Stim Bias Contra', 'red')]
        title_str = 'Reward-aligned: Stim Bias Ipsi/Contra (mean ± SEM across sessions)'
        save_name = 'reward_aligned_stim_bias_ipsi_contra_diff_across_sessions'
    else:
        raise ValueError("key_type must be 'hemi' or 'bias'")

    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Dictionary to save the difference curve of each session
    diff_traces = {cond[0]: [] for cond in stim_conditions}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            try:
                with open(pkl_file, 'rb') as f:
                    dffTrace_mean_reward, _, _ = pickle.load(f)
                for condition, _ in stim_conditions:
                    trace_rewarded = dffTrace_mean_reward.get(condition)
                    trace_unrewarded = dffTrace_mean_reward.get('Unrewarded')
                    if trace_rewarded is not None and trace_unrewarded is not None:
                        mean_rewarded = np.nanmean(trace_rewarded, axis=0)
                        mean_unrewarded = np.nanmean(trace_unrewarded, axis=0)
                        diff_trace = mean_rewarded - mean_unrewarded
                        diff_traces[condition].append(diff_trace)
            except Exception as e:
                print(f"Error processing session {getattr(recordingList, 'sessionName', [''])[ind]}: {str(e)}")
                continue

    plt.figure(figsize=(7, 4))
    for condition, color in stim_conditions:
        traces = np.array(diff_traces[condition])
        if traces.shape[0] > 0:
            mean_across_sessions = np.nanmean(traces, axis=0)
            sem_across_sessions = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
            plt.plot(time_axis[:len(mean_across_sessions)], mean_across_sessions, color=color, label=condition + ' (Rewarded - Unrewarded)', linewidth=2)
            plt.fill_between(
                time_axis[:len(mean_across_sessions)],
                mean_across_sessions - sem_across_sessions,
                mean_across_sessions + sem_across_sessions,
                color=color, alpha=0.3
            )
    plt.axvline(0, color='k', linestyle=':', linewidth=1)
    plt.xlabel('Time from reward (s)')
    plt.ylabel('df/f (Rewarded - Unrewarded)')
    plt.title(title_str)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(save_name, save_path)
    plt.close()

def plot_reward_aligned_stim_ipsi_contra_diff_scatter_window_across_sessions(
    recordingList,
    time_window=[0.1, 0.8],
    subfolder='responsive_neurons',
    save_path=None,
    key_type='hemi'  # 'hemi' (default) or 'bias'
):
    """
    For each session and condition (Stim Hemi Ipsi/Contra or Stim Bias Ipsi/Contra), calculate the mean of the curve (rewarded-unrewarded)
    in the specified time window and plot:
    - Individual points of each session
    - Mean and SEM as errorbar
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from scipy.stats import ttest_ind

    if key_type == 'hemi':
        stim_conditions = [('Stim Hemi Ipsi', 'blue'), ('Stim Hemi Contra', 'red')]
        labels = ['Stim Hemi Ipsi', 'Stim Hemi Contra']
        title_str = 'Reward-aligned: Stim Hemi Ipsi/Contra (window mean across sessions)'
        save_name = 'reward_aligned_stim_hemi_ipsi_contra_diff_scatter_window'
    elif key_type == 'bias':
        stim_conditions = [('Stim Bias Ipsi', 'blue'), ('Stim Bias Contra', 'red')]
        labels = ['Stim Bias Ipsi', 'Stim Bias Contra']
        title_str = 'Reward-aligned: Stim Bias Ipsi/Contra (window mean across sessions)'
        save_name = 'reward_aligned_stim_bias_ipsi_contra_diff_scatter_window'
    else:
        raise ValueError("key_type must be 'hemi' or 'bias'")

    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Calculate the indices of the time window
    start_idx = int((time_window[0] + pre_stim_sec) * fRate)
    end_idx = int((time_window[1] + pre_stim_sec) * fRate)

    # Dictionary to save the mean of the window of each session
    window_means = {cond[0]: [] for cond in stim_conditions}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            try:
                with open(pkl_file, 'rb') as f:
                    dffTrace_mean_reward, _, _ = pickle.load(f)
                for condition, _ in stim_conditions:
                    trace_rewarded = dffTrace_mean_reward.get(condition)
                    trace_unrewarded = dffTrace_mean_reward.get('Unrewarded')
                    if trace_rewarded is not None and trace_unrewarded is not None:
                        mean_rewarded = np.nanmean(trace_rewarded, axis=0)
                        mean_unrewarded = np.nanmean(trace_unrewarded, axis=0)
                        diff_trace = mean_rewarded - mean_unrewarded
                        # Mean in the time window
                        window_mean = np.nanmean(diff_trace[start_idx:end_idx])
                        window_means[condition].append(window_mean)
            except Exception as e:
                print(f"Error processing session {getattr(recordingList, 'sessionName', [''])[ind]}: {str(e)}")
                continue

    # Prepare data for the scatter and errorbar
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = [0, 1]
    colors = [c[1] for c in stim_conditions]
    all_means = []
    all_sems = []
    all_vals = []
    for i, cond in enumerate(labels):
        vals = window_means[cond]
        # Puntos individuales
        ax.scatter([xvals[i]]*len(vals), vals, color=colors[i], alpha=0.7, s=60, edgecolor='k', label=f'{cond} (sessions)')
        # Mean and SEM
        mean = np.nanmean(vals) if len(vals) > 0 else np.nan
        sem = np.nanstd(vals)/np.sqrt(len(vals)) if len(vals) > 0 else np.nan
        all_means.append(mean)
        all_sems.append(sem)
        all_vals.append(vals)
    # Errorbar
    ax.errorbar(xvals, all_means, yerr=all_sems, fmt='o', color='k', capsize=8, markersize=12, label='Mean ± SEM')
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(f"Mean df/f (Rewarded - Unrewarded)\n[{time_window[0]}, {time_window[1]}] s")
    # Student's t-test
    if len(all_vals[0]) > 1 and len(all_vals[1]) > 1:
        tstat, pval = ttest_ind(all_vals[0], all_vals[1], nan_policy='omit')
        print(f"T-test {labels[0]} vs {labels[1]}: t = {tstat:.3f}, p = {pval:.4g}")
        pval_str = f"p = {pval:.3g}"
    else:
        pval_str = "p = N/A"
    ax.set_title(f'{title_str}\nT-test: {pval_str}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.25, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'{save_name}_{time_window[0]}_{time_window[1]}', save_path)
    plt.close()


def plot_aligned_by_contrasts(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions=None,
    alignment='reward',  # 'reward', 'choice' or 'stimulus'
    baseline_window=[-0.2, 0]  # in seconds
):
    """
    Plot neural activity aligned to reward, choice, or stimulus for different contrasts, combining both sides (Left and Right).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import seaborn as sns

    print(f"Starting {alignment}-aligned activity by combined contrasts for all sessions...")

    colormap = 'viridis'
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            session_name = recordingList.sessionName.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            print(f'Plotting {alignment}-aligned activity for session: {session_name}')
            try:
                # Cargar datos
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)

                # Seleccionar diccionario correcto
                if alignment == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment in ['stimulus', 'stim']:
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    print(f"Unknown alignment: {alignment}")
                    continue

                # Default conditions
                if contrast_conditions is None:
                    if alignment == 'reward':
                        contrast_conditions = [
                             '-0.0625 Rewarded','-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded', '0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
                        ]
                    elif alignment in ['stimulus', 'stim']:
                        contrast_conditions = [
                             '-0.0625 Rewarded','-0.125 Rewarded', '-0.25 Rewarded', '-0.5 Rewarded', '0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
                        ]
                    else:  # choice
                        contrast_conditions = [
                            '0 Choice', '0.0625 Choice', '0.125 Choice', '0.25 Choice', '0.5 Choice'
                        ]

                # Time axis
                fRate = 30
                pre_stim_sec = 2
                total_time = 8
                n_frames = int(total_time * fRate)
                time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                colors = sns.color_palette(colormap, len(contrast_conditions))[::-1]

                for i, contrast_cond in enumerate(contrast_conditions):
                    data = dffTrace_mean.get(contrast_cond)
                    if data is not None and np.size(data) > 0:
                        mean_trace = np.nanmean(data, axis=0)
                        if baseline_window is not None:
                            baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                            baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                            if baseline_end_idx == baseline_start_idx:
                                baseline_end_idx += 1
                            baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                        else:
                            baseline_frames = int(0.2 * fRate)
                            baseline = np.nanmean(mean_trace[fRate*2 - baseline_frames:fRate*2])
                        mean_trace_bs = mean_trace - baseline
                        sem_trace = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
                        sem_trace_bs = sem_trace
                        ax.plot(time_axis[:len(mean_trace_bs)], mean_trace_bs,
                                color=colors[i], label=contrast_cond, linewidth=2)
                        ax.fill_between(
                            time_axis[:len(mean_trace_bs)],
                            mean_trace_bs - sem_trace_bs,
                            mean_trace_bs + sem_trace_bs,
                            color=colors[i], alpha=0.3
                        )

                ax.axvline(0, color='k', linestyle=':', linewidth=1)
                ax.set_xlabel(f'Time from {alignment.capitalize()} (s)')
                ax.set_ylabel('df/f')
                ax.set_title(f'{session_name} - Calcium activity by contrast')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=14)
                plt.tight_layout()
                if save_path is not None:
                    save_figureAll(f'{session_name}_{alignment}_aligned_by_contrasts', save_path)
                plt.close()
                print(f"✓ Successfully plotted and saved for session: {session_name}")
            except Exception as e:
                print(f"✗ Error plotting session {session_name}: {str(e)}")
                continue
    print(f"Completed {alignment}-aligned activity by combined contrasts for all sessions.")

def plot_peak_window_mean_dff_by_contrast_hemi_ipsi_contra(
    recordingList,
    event_type='stimulus',
    save_path=None,
    title=None,
    subfolder='responsive_neurons',
    use_zscored=True,
    baseline_window=[-0.2, 0],
    contrast_conditions_ipsi=None,
    contrast_conditions_contra=None
):
    """
    Plot mean df/f for a window of ±250 ms around the peak of the mean curve for each contrast and session, separated by ipsi/contra.
    Shows Ipsi and Contra in separate subplots (columns), sharing the Y axis.
    Allows passing the contrast conditions as arguments.
    No legend for individual scatter points.
    Colors go from lighter (lower contrast) to darker (higher contrast).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    # Time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before the stimulus/event
    total_time = 8  # seconds (for the time axis)
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Default conditions if not provided
    if contrast_conditions_ipsi is None:
        contrast_conditions_ipsi = ['-0.0625 Rewarded Hemi Ipsi', '-0.125 Rewarded Hemi Ipsi', '-0.25 Rewarded Hemi Ipsi', '-0.5 Rewarded Hemi Ipsi', '0 Rewarded Hemi Ipsi', '0.0625 Rewarded Hemi Ipsi', '0.125 Rewarded Hemi Ipsi', '0.25 Rewarded Hemi Ipsi', '0.5 Rewarded Hemi Ipsi']
    if contrast_conditions_contra is None:
        contrast_conditions_contra = ['0.0625 Rewarded Hemi Contra', '0.125 Rewarded Hemi Contra', '0.25 Rewarded Hemi Contra', '0.5 Rewarded Hemi Contra', '0 Rewarded Hemi Contra', '0.0625 Rewarded Hemi Contra', '0.125 Rewarded Hemi Contra', '0.25 Rewarded Hemi Contra', '0.5 Rewarded Hemi Contra']
    contrast_values_ipsi = [-0.0625, -0.125, -0.25, -0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    contrast_values_contra = [0.0625, 0.125, 0.25, 0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    # Colors: lighter for lower contrast, darker for higher contrast
    colors_ipsi = sns.color_palette('Blues', len(contrast_conditions_ipsi))  # light to dark
    colors_contra = sns.color_palette('Reds', len(contrast_conditions_contra))  # light to dark

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_ipsi, ax_contra = axes

    contrast_data_ipsi = {contrast: [] for contrast in contrast_conditions_ipsi}
    contrast_data_contra = {contrast: [] for contrast in contrast_conditions_contra}
    session_points_ipsi = {}
    session_points_contra = {}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                # IPSI
                session_contrasts_ipsi = []
                session_means_ipsi = []
                for i, contrast in enumerate(contrast_conditions_ipsi):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        data = dffTrace_mean[contrast]
                        if data.shape[1] > 0:
                            mean_trace = np.nanmean(data, axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            mean_trace_bs = mean_trace - baseline
                            peak_window = (time_axis >= 0) & (time_axis <= 2)
                            peak_idx = np.argmax(np.abs(mean_trace_bs[peak_window])) + np.where(peak_window)[0][0]
                            window_size = int(0.25 * fRate_imaging)
                            start_idx = max(0, peak_idx - window_size)
                            end_idx = min(len(mean_trace_bs), peak_idx + window_size)
                            dff_bs = data - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_idx:end_idx])
                            contrast_data_ipsi[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_ipsi[i]
                            })
                            session_contrasts_ipsi.append(contrast_values_ipsi[i])
                            session_means_ipsi.append(mean_dff)
                if len(session_contrasts_ipsi) > 1:
                    session_points_ipsi[session_name] = (session_contrasts_ipsi, session_means_ipsi)
                # CONTRA
                session_contrasts_contra = []
                session_means_contra = []
                for i, contrast in enumerate(contrast_conditions_contra):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        data = dffTrace_mean[contrast]
                        if data.shape[1] > 0:
                            mean_trace = np.nanmean(data, axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            mean_trace_bs = mean_trace - baseline
                            peak_window = (time_axis >= 0) & (time_axis <= 2)
                            peak_idx = np.argmax(np.abs(mean_trace_bs[peak_window])) + np.where(peak_window)[0][0]
                            window_size = int(0.25 * fRate_imaging)
                            start_idx = max(0, peak_idx - window_size)
                            end_idx = min(len(mean_trace_bs), peak_idx + window_size)
                            dff_bs = data - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_idx:end_idx])
                            contrast_data_contra[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_contra[i]
                            })
                            session_contrasts_contra.append(contrast_values_contra[i])
                            session_means_contra.append(mean_dff)
                if len(session_contrasts_contra) > 1:
                    session_points_contra[session_name] = (session_contrasts_contra, session_means_contra)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue
    # IPSI subplot
    for session_name, (xvals, yvals) in session_points_ipsi.items():
        ax_ipsi.plot(xvals, yvals, linestyle='--', color='blue', alpha=0.5, linewidth=1, zorder=1)
    for i, contrast in enumerate(contrast_conditions_ipsi):
        if contrast_data_ipsi[contrast]:
            df_contrast = pd.DataFrame(contrast_data_ipsi[contrast])
            ax_ipsi.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                            color=colors_ipsi[i], alpha=0.7, s=50,
                            label=None, zorder=2)
    means_ipsi = []
    sems_ipsi = []
    for i, contrast in enumerate(contrast_conditions_ipsi):
        vals = [d['mean_dff'] for d in contrast_data_ipsi[contrast]]
        if len(vals) > 0:
            means_ipsi.append(np.mean(vals))
            sems_ipsi.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_ipsi.append(np.nan)
            sems_ipsi.append(np.nan)
    for i, (x, m, s) in enumerate(zip(contrast_values_ipsi, means_ipsi, sems_ipsi)):
        if not np.isnan(m):
            ax_ipsi.errorbar(x, m, yerr=s, fmt='_', color=colors_ipsi[i], elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    ax_ipsi.set_xlabel('Contrast')
    ax_ipsi.set_ylabel(f'Mean df/f (peak ±250ms)')
    ax_ipsi.set_title('Ipsi')
    ax_ipsi.set_xscale('linear')
    ax_ipsi.set_xticks(contrast_values_ipsi)
    ax_ipsi.set_xticklabels([str(c) for c in contrast_values_ipsi])
    import matplotlib.pyplot as plt
    plt.setp(ax_ipsi.get_xticklabels(), rotation=45, ha='right')
    ax_ipsi.grid(True, alpha=0.3)
    # CONTRA subplot
    for session_name, (xvals, yvals) in session_points_contra.items():
        ax_contra.plot(xvals, yvals, linestyle='--', color='red', alpha=0.5, linewidth=1, zorder=1)
    for i, contrast in enumerate(contrast_conditions_contra):
        if contrast_data_contra[contrast]:
            df_contrast = pd.DataFrame(contrast_data_contra[contrast])
            ax_contra.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                              color=colors_contra[i], alpha=0.7, s=50,
                              label=None, zorder=2)
    means_contra = []
    sems_contra = []
    for i, contrast in enumerate(contrast_conditions_contra):
        vals = [d['mean_dff'] for d in contrast_data_contra[contrast]]
        if len(vals) > 0:
            means_contra.append(np.mean(vals))
            sems_contra.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_contra.append(np.nan)
            sems_contra.append(np.nan)
    for i, (x, m, s) in enumerate(zip(contrast_values_contra, means_contra, sems_contra)):
        if not np.isnan(m):
            ax_contra.errorbar(x, m, yerr=s, fmt='_', color=colors_contra[i], elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    ax_contra.set_xlabel('Contrast')
    ax_contra.set_title('Contra')
    ax_contra.set_xscale('linear')
    ax_contra.set_xticks(contrast_values_contra)
    ax_contra.set_xticklabels([str(c) for c in contrast_values_contra])
    plt.setp(ax_contra.get_xticklabels(), rotation=45, ha='right')
    ax_contra.grid(True, alpha=0.3)
    # General title
    if title is None:
        title = f'Mean df/f by contrast (±250 ms around peak, baseline subtracted) - Ipsi vs Contra - {subfolder}'
    fig.suptitle(title, y=1.02, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        save_figureAll(f'peak_window_mean_dff_by_contrast_hemi_ipsi_contra_{event_type}_{subfolder}_baseline', save_path)
    plt.close(fig)
    return fig

def plot_peak_window_mean_dff_by_contrast_bias_ipsi_contra(
    recordingList,
    event_type='stimulus',
    save_path=None,
    title='Mean df/f by contrast (±250 ms around peak, baseline subtracted) - Rewarded trials',
    subfolder='responsive_neurons',
    use_zscored=True,
    baseline_window=[-0.2, 0],
    contrast_conditions_bias=None,
    contrast_conditions_nobias=None
):
    """
    Plot mean df/f for a window of ±250 ms around the peak of the mean curve for each contrast and session, separated by bias-ipsi and bias-contra.
    Shows bias-ipsi and bias-contra in separate subplots (columns), sharing the Y axis.
    No legend for individual scatter points.
    Colors go from lighter (lower contrast) to darker (higher contrast).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    # Time parameters
    fRate_imaging = 30  # Hz
    pre_stim_sec = 2  # seconds before the stimulus/event
    total_time = 8  # seconds (for the time axis)
    n_frames = int(total_time * fRate_imaging)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    # Default conditions if not provided
    if contrast_conditions_bias is None:
        contrast_conditions_bias = ['-0.0625 Rewarded Bias Ipsi', '-0.125 Rewarded Bias Ipsi', '-0.25 Rewarded Bias Ipsi', '-0.5 Rewarded Bias Ipsi', '0 Rewarded Bias Ipsi', '0.0625 Rewarded Bias Ipsi', '0.125 Rewarded Bias Ipsi', '0.25 Rewarded Bias Ipsi', '0.5 Rewarded Bias Ipsi']
    if contrast_conditions_nobias is None:
        contrast_conditions_nobias = ['0.0625 Rewarded Bias Contra', '0.125 Rewarded Bias Contra', '0.25 Rewarded Bias Contra', '0.5 Rewarded Bias Contra', '0 Rewarded Bias Contra', '0.0625 Rewarded Bias Contra', '0.125 Rewarded Bias Contra', '0.25 Rewarded Bias Contra', '0.5 Rewarded Bias Contra']
    contrast_values_bias = [-0.0625, -0.125, -0.25, -0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    contrast_values_nobias = [0.0625, 0.125, 0.25, 0.5, 0, 0.0625, 0.125, 0.25, 0.5]
    colors_bias = sns.color_palette('Greens', len(contrast_conditions_bias))  # light to dark
    colors_nobias = sns.color_palette('Purples', len(contrast_conditions_nobias))  # light to dark

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_bias, ax_nobias = axes

    contrast_data_bias = {contrast: [] for contrast in contrast_conditions_bias}
    contrast_data_nobias = {contrast: [] for contrast in contrast_conditions_nobias}
    session_points_bias = {}
    session_points_nobias = {}

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                # BIAS IPSI
                session_contrasts_bias = []
                session_means_bias = []
                for i, contrast in enumerate(contrast_conditions_bias):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        data = dffTrace_mean[contrast]
                        if data.shape[1] > 0:
                            mean_trace = np.nanmean(data, axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            mean_trace_bs = mean_trace - baseline
                            peak_window = (time_axis >= 0) & (time_axis <= 2)
                            peak_idx = np.argmax(np.abs(mean_trace_bs[peak_window])) + np.where(peak_window)[0][0]
                            window_size = int(0.25 * fRate_imaging)
                            start_idx = max(0, peak_idx - window_size)
                            end_idx = min(len(mean_trace_bs), peak_idx + window_size)
                            dff_bs = data - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_idx:end_idx])
                            contrast_data_bias[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_bias[i]
                            })
                            session_contrasts_bias.append(contrast_values_bias[i])
                            session_means_bias.append(mean_dff)
                if len(session_contrasts_bias) > 1:
                    session_points_bias[session_name] = (session_contrasts_bias, session_means_bias)
                # BIAS CONTRA
                session_contrasts_nobias = []
                session_means_nobias = []
                for i, contrast in enumerate(contrast_conditions_nobias):
                    if contrast in dffTrace_mean and dffTrace_mean[contrast] is not None:
                        data = dffTrace_mean[contrast]
                        if data.shape[1] > 0:
                            mean_trace = np.nanmean(data, axis=0)
                            if baseline_window is not None:
                                baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                                baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                                if baseline_end_idx == baseline_start_idx:
                                    baseline_end_idx += 1
                                baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                            else:
                                baseline_frames = int(0.2 * fRate_imaging)
                                baseline = np.nanmean(mean_trace[fRate_imaging*2 - baseline_frames:fRate_imaging*2])
                            mean_trace_bs = mean_trace - baseline
                            peak_window = (time_axis >= 0) & (time_axis <= 2)
                            peak_idx = np.argmax(np.abs(mean_trace_bs[peak_window])) + np.where(peak_window)[0][0]
                            window_size = int(0.25 * fRate_imaging)
                            start_idx = max(0, peak_idx - window_size)
                            end_idx = min(len(mean_trace_bs), peak_idx + window_size)
                            dff_bs = data - baseline
                            mean_dff = np.nanmean(dff_bs[:, start_idx:end_idx])
                            contrast_data_nobias[contrast].append({
                                'animal': animal_id,
                                'session': session_name,
                                'mean_dff': mean_dff,
                                'contrast': contrast_values_nobias[i]
                            })
                            session_contrasts_nobias.append(contrast_values_nobias[i])
                            session_means_nobias.append(mean_dff)
                if len(session_contrasts_nobias) > 1:
                    session_points_nobias[session_name] = (session_contrasts_nobias, session_means_nobias)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue
    # BIAS IPSI subplot
    for session_name, (xvals, yvals) in session_points_bias.items():
        ax_bias.plot(xvals, yvals, linestyle='--', color='green', alpha=0.5, linewidth=1, zorder=1)
    for i, contrast in enumerate(contrast_conditions_bias):
        if contrast_data_bias[contrast]:
            df_contrast = pd.DataFrame(contrast_data_bias[contrast])
            ax_bias.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                            color=colors_bias[i], alpha=0.7, s=50,
                            label=None, zorder=2)
    means_bias = []
    sems_bias = []
    for i, contrast in enumerate(contrast_conditions_bias):
        vals = [d['mean_dff'] for d in contrast_data_bias[contrast]]
        if len(vals) > 0:
            means_bias.append(np.mean(vals))
            sems_bias.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_bias.append(np.nan)
            sems_bias.append(np.nan)
    for i, (x, m, s) in enumerate(zip(contrast_values_bias, means_bias, sems_bias)):
        if not np.isnan(m):
            ax_bias.errorbar(x, m, yerr=s, fmt='_', color=colors_bias[i], elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    ax_bias.set_xlabel('Contrast')
    ax_bias.set_ylabel(f'Mean df/f (peak ±250ms)')
    ax_bias.set_title('Bias Ipsi')
    ax_bias.set_xscale('linear')
    ax_bias.set_xticks(contrast_values_bias)
    ax_bias.set_xticklabels([str(c) for c in contrast_values_bias])
    import matplotlib.pyplot as plt
    plt.setp(ax_bias.get_xticklabels(), rotation=45, ha='right')
    ax_bias.grid(True, alpha=0.3)
    # BIAS CONTRA subplot
    for session_name, (xvals, yvals) in session_points_nobias.items():
        ax_nobias.plot(xvals, yvals, linestyle='--', color='purple', alpha=0.5, linewidth=1, zorder=1)
    for i, contrast in enumerate(contrast_conditions_nobias):
        if contrast_data_nobias[contrast]:
            df_contrast = pd.DataFrame(contrast_data_nobias[contrast])
            ax_nobias.scatter(df_contrast['contrast'], df_contrast['mean_dff'],
                              color=colors_nobias[i], alpha=0.7, s=50,
                              label=None, zorder=2)
    means_nobias = []
    sems_nobias = []
    for i, contrast in enumerate(contrast_conditions_nobias):
        vals = [d['mean_dff'] for d in contrast_data_nobias[contrast]]
        if len(vals) > 0:
            means_nobias.append(np.mean(vals))
            sems_nobias.append(np.std(vals)/np.sqrt(len(vals)))
        else:
            means_nobias.append(np.nan)
            sems_nobias.append(np.nan)
    for i, (x, m, s) in enumerate(zip(contrast_values_nobias, means_nobias, sems_nobias)):
        if not np.isnan(m):
            ax_nobias.errorbar(x, m, yerr=s, fmt='_', color=colors_nobias[i], elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
    ax_nobias.set_xlabel('Contrast')
    ax_nobias.set_title('Bias Contra')
    ax_nobias.set_xscale('linear')
    ax_nobias.set_xticks(contrast_values_nobias)
    ax_nobias.set_xticklabels([str(c) for c in contrast_values_nobias])
    plt.setp(ax_nobias.get_xticklabels(), rotation=45, ha='right')
    ax_nobias.grid(True, alpha=0.3)
    # General title
    if title is None:
        title = f'Mean df/f by contrast (±250 ms around peak, baseline subtracted) - Bias Ipsi vs Bias Contra - {subfolder}'
    fig.suptitle(title, y=1.02, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        save_figureAll(f'peak_window_mean_dff_by_contrast_bias_ipsi_contra_{event_type}_{subfolder}_baseline', save_path)
    plt.close(fig)
    return fig

def plot_event_aligned_by_bias_type_across_sessions(
    recordingList,
    bias_type='Choice Bias',
    alignment='choice',
    subfolder='responsive_neurons',
    save_path=None,
    title=None,
    by_contrast=False
):
    """
    Plot the average activity (mean ± SEM) of responsive neurons aligned to the specified event (reward, stimulus, or choice),
    separated by bias_type ('Choice Bias Ipsi' vs 'Choice Bias Contra' or 'Stim Bias Ipsi' vs 'Stim Bias Contra'), averaged across sessions.
    The legend shows the total number of trials of each type across all sessions.
    If by_contrast=True, creates a subplot for each contrast (e.g. '0.0625 Rewarded Bias Ipsi', etc.), ordered from lowest to highest.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os

    # Time axis (assuming -2 to 6 s, 30 Hz)
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    if not by_contrast:
        # --- Classic mode (no contrast separation) ---
        if bias_type.lower().startswith('choice'):
            key_ipsi = 'Choice Bias Ipsi'
            key_contra = 'Choice Bias Contra'
        elif bias_type.lower().startswith('stim'):
            key_ipsi = 'Stim Bias Ipsi'
            key_contra = 'Stim Bias Contra'
        else:
            print("bias_type must be 'Choice Bias' or 'Stim Bias'")
            return
        all_ipsi = []
        all_contra = []
        n_trials_ipsi = 0
        n_trials_contra = 0
        for ind, recordingDate in enumerate(recordingList.recordingDate):
            if recordingList.imagingDataExtracted[ind] == 1:
                session_name = recordingList.sessionName[ind]
                pathname = recordingList.analysispathname[ind]
                subfolder_path = os.path.join(pathname, subfolder)
                pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
                if not os.path.exists(pkl_file):
                    continue
                with open(pkl_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                if alignment.lower() == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment.lower() == 'stimulus':
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment.lower() == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    print(f"alignment must be 'reward', 'stimulus' or 'choice' (in session {session_name})")
                    continue
                data_ipsi = dffTrace_mean.get(key_ipsi)
                data_contra = dffTrace_mean.get(key_contra)
                if data_ipsi is not None:
                    mean_ipsi = np.nanmean(data_ipsi, axis=0)
                    all_ipsi.append(mean_ipsi)
                    n_trials_ipsi += data_ipsi.shape[0]
                if data_contra is not None:
                    mean_contra = np.nanmean(data_contra, axis=0)
                    all_contra.append(mean_contra)
                    n_trials_contra += data_contra.shape[0]
        all_ipsi = np.array(all_ipsi)
        all_contra = np.array(all_contra)
        mean_ipsi = np.nanmean(all_ipsi, axis=0) if all_ipsi.size > 0 else np.full(n_frames, np.nan)
        sem_ipsi = np.nanstd(all_ipsi, axis=0) / np.sqrt(all_ipsi.shape[0]) if all_ipsi.shape[0] > 1 else np.full(n_frames, np.nan)
        mean_contra = np.nanmean(all_contra, axis=0) if all_contra.size > 0 else np.full(n_frames, np.nan)
        sem_contra = np.nanstd(all_contra, axis=0) / np.sqrt(all_contra.shape[0]) if all_contra.shape[0] > 1 else np.full(n_frames, np.nan)
        plt.figure(figsize=(8, 4))
        plt.plot(time_axis[:len(mean_ipsi)], mean_ipsi, label=f'Ipsi (n={n_trials_ipsi})', color='royalblue')
        plt.fill_between(time_axis[:len(mean_ipsi)], mean_ipsi - sem_ipsi, mean_ipsi + sem_ipsi, color='royalblue', alpha=0.3)
        plt.plot(time_axis[:len(mean_contra)], mean_contra, label=f'Contra (n={n_trials_contra})', color='orange')
        plt.fill_between(time_axis[:len(mean_contra)], mean_contra - sem_contra, mean_contra + sem_contra, color='orange', alpha=0.3)
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel(f'Time from {alignment.lower()} (s)')
        plt.ylabel('dF/F')
        if title is None:
            title = f'{alignment.capitalize()}-aligned neural activity by {bias_type} (Ipsi vs Contra, mean ± SEM across sessions)'
        plt.title(title)
        plt.legend(loc='best', fontsize=10, frameon=True)
        plt.tight_layout()
        if save_path is not None:
            save_figureAll(f'{alignment}_aligned_by_{bias_type.replace(" ", "")}_ipsi_contra_across_sessions', save_path)
        plt.close()
        return
    # --- Contrast mode ---
    # Use only the specified contrast keys
    contrast_keys = ['-0.0625 Rewarded Bias Ipsi', '-0.125 Rewarded Bias Ipsi', '-0.25 Rewarded Bias Ipsi', '-0.5 Rewarded Bias Ipsi', '0 Rewarded Bias Ipsi', '0.0625 Rewarded Bias Ipsi', '0.125 Rewarded Bias Ipsi', '0.25 Rewarded Bias Ipsi', '0.5 Rewarded Bias Ipsi']
    contrast_vals = [-0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5]
    n_contrasts = len(contrast_keys)
    fig, axes = plt.subplots(1, n_contrasts, figsize=(5*n_contrasts, 4), sharey=True)
    if n_contrasts == 1:
        axes = [axes]
    for idx, (contrast, contrast_val) in enumerate(zip(contrast_keys, contrast_vals)):
        all_ipsi = []
        all_contra = []
        n_trials_ipsi = 0
        n_trials_contra = 0
        key_ipsi = contrast
        key_contra = contrast.replace('Ipsi', 'Contra')
        for ind, recordingDate in enumerate(recordingList.recordingDate):
            if recordingList.imagingDataExtracted[ind] == 1:
                session_name = recordingList.sessionName[ind]
                pathname = recordingList.analysispathname[ind]
                subfolder_path = os.path.join(pathname, subfolder)
                pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
                if not os.path.exists(pkl_file):
                    continue
                with open(pkl_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                if alignment.lower() == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment.lower() == 'stimulus':
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment.lower() == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    continue
                data_ipsi = dffTrace_mean.get(key_ipsi)
                data_contra = dffTrace_mean.get(key_contra)
                if data_ipsi is not None:
                    mean_ipsi = np.nanmean(data_ipsi, axis=0)
                    all_ipsi.append(mean_ipsi)
                    n_trials_ipsi += data_ipsi.shape[0]
                if data_contra is not None:
                    mean_contra = np.nanmean(data_contra, axis=0)
                    all_contra.append(mean_contra)
                    n_trials_contra += data_contra.shape[0]
        all_ipsi = np.array(all_ipsi)
        all_contra = np.array(all_contra)
        mean_ipsi = np.nanmean(all_ipsi, axis=0) if all_ipsi.size > 0 else np.full(n_frames, np.nan)
        sem_ipsi = np.nanstd(all_ipsi, axis=0) / np.sqrt(all_ipsi.shape[0]) if all_ipsi.shape[0] > 1 else np.full(n_frames, np.nan)
        mean_contra = np.nanmean(all_contra, axis=0) if all_contra.size > 0 else np.full(n_frames, np.nan)
        sem_contra = np.nanstd(all_contra, axis=0) / np.sqrt(all_contra.shape[0]) if all_contra.shape[0] > 1 else np.full(n_frames, np.nan)
        ax = axes[idx]
        ax.plot(time_axis[:len(mean_ipsi)], mean_ipsi, label=f'Ipsi (n={n_trials_ipsi})', color='royalblue')
        ax.fill_between(time_axis[:len(mean_ipsi)], mean_ipsi - sem_ipsi, mean_ipsi + sem_ipsi, color='royalblue', alpha=0.3)
        ax.plot(time_axis[:len(mean_contra)], mean_contra, label=f'Contra (n={n_trials_contra})', color='orange')
        ax.fill_between(time_axis[:len(mean_contra)], mean_contra - sem_contra, mean_contra + sem_contra, color='orange', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'Time from {alignment.lower()} (s)')
        if idx == 0:
            ax.set_ylabel('dF/F')
        ax.set_title(f'Contrast {contrast_val}')
        ax.legend(loc='best', fontsize=10, frameon=True)
    if title is None:
        title = f'{alignment.capitalize()}-aligned neural activity by {bias_type} (Ipsi vs Contra, mean ± SEM across sessions)'
    fig.suptitle(title, y=1.04, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        save_figureAll(f'{alignment}_aligned_by_{bias_type.replace(" ", "")}_ipsi_contra_across_sessions_by_contrast', save_path)
    plt.close(fig)

def plot_event_aligned_bias_diff_scatter_window_across_sessions(
    recordingList,
    time_window=[0.1, 0.8],
    bias_type='Choice Bias',
    alignment='choice',
    subfolder='responsive_neurons',
    save_path=None
):
    """
    For each session and condition (Ipsi and Contra), calculates the mean of the curve in the specified time window and plots:
    - Individual points of each session
    - Mean and SEM as errorbar
    - Performs a statistical test between both groups
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from scipy.stats import ttest_ind

    # Time axis (assuming -2 to 6 s, 30 Hz)
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
    start_idx = int((time_window[0] + pre_stim_sec) * fRate)
    end_idx = int((time_window[1] + pre_stim_sec) * fRate)

    # Determine the key pair according to bias_type
    if bias_type.lower().startswith('choice'):
        key_ipsi = 'Choice Bias Ipsi'
        key_contra = 'Choice Bias Contra'
    elif bias_type.lower().startswith('stim'):
        key_ipsi = 'Stim Bias Ipsi'
        key_contra = 'Stim Bias Contra'
    else:
        print("bias_type must be 'Choice Bias' or 'Stim Bias'")
        return

    # Accumulate the mean of the window for each session
    window_means = {'Ipsi': [], 'Contra': []}
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
            # Select the dictionary according to alignment
            if alignment.lower() == 'reward':
                dffTrace_mean = dffTrace_mean_reward
            elif alignment.lower() == 'stimulus':
                dffTrace_mean = dffTrace_mean_stimuli
            elif alignment.lower() == 'choice':
                dffTrace_mean = dffTrace_mean_choice
            else:
                print(f"alignment must be 'reward', 'stimulus' or 'choice' (in session {session_name})")
                continue
            data_ipsi = dffTrace_mean.get(key_ipsi)
            data_contra = dffTrace_mean.get(key_contra)
            if data_ipsi is not None:
                mean_ipsi = np.nanmean(data_ipsi, axis=0)
                window_mean_ipsi = np.nanmean(mean_ipsi[start_idx:end_idx])
                window_means['Ipsi'].append(window_mean_ipsi)
            if data_contra is not None:
                mean_contra = np.nanmean(data_contra, axis=0)
                window_mean_contra = np.nanmean(mean_contra[start_idx:end_idx])
                window_means['Contra'].append(window_mean_contra)

    # Prepare data for the scatter and errorbar
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = [0, 1]
    colors = ['royalblue', 'orange']
    labels = ['Ipsi', 'Contra']
    all_means = []
    all_sems = []
    all_vals = []
    for i, cond in enumerate(labels):
        vals = window_means[cond]
        # Individual points
        ax.scatter([xvals[i]]*len(vals), vals, color=colors[i], alpha=0.7, s=60, edgecolor='k', label=f'{cond} (sessions)')
        # Mean and SEM
        mean = np.nanmean(vals) if len(vals) > 0 else np.nan
        sem = np.nanstd(vals)/np.sqrt(len(vals)) if len(vals) > 0 else np.nan
        all_means.append(mean)
        all_sems.append(sem)
        all_vals.append(vals)
    # Errorbar
    ax.errorbar(xvals, all_means, yerr=all_sems, fmt='o', color='k', capsize=8, markersize=12, label='Mean ± SEM')
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(f"Mean df/f (Rewarded - Unrewarded)\n[{time_window[0]}, {time_window[1]}] s")
    # Student's t-test
    if len(all_vals[0]) > 1 and len(all_vals[1]) > 1:
        tstat, pval = ttest_ind(all_vals[0], all_vals[1], nan_policy='omit')
        print(f"T-test Ipsi vs Contra: t = {tstat:.3f}, p = {pval:.4g}")
        pval_str = f"p = {pval:.3g}"
    else:
        pval_str = "p = N/A"
    ax.set_title(f'{alignment.capitalize()}-aligned: {bias_type} (window mean across sessions)\nT-test: {pval_str}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.25, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'{alignment}_aligned_{bias_type.replace(" ", "")}_ipsi_contra_diff_scatter_window_{time_window[0]}_{time_window[1]}', save_path)
    plt.close()

def plot_mean_sem_across_sessions_by_contrasts(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None,
    contrast_conditions=None,
    alignment='reward',  # 'reward', 'choice' or 'stimulus'
    baseline_window=[-0.2, 0],  # in seconds
    title=None
):
    """
    Plot the mean and SEM across sessions for each contrast condition, all in a single plot.
    Each curve is the mean trace (with SEM band) across sessions for a given contrast.
    Baseline subtraction is performed using the baseline_window (in seconds) if provided, otherwise the last 200 ms before the event is used by default.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import seaborn as sns

    # Default contrast conditions if not provided
    if contrast_conditions is None:
        if alignment == 'reward':
            contrast_conditions = [
                '0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
            ]
        elif alignment in ['stimulus', 'stim']:
            contrast_conditions = [
                '0 Rewarded', '0.0625 Rewarded', '0.125 Rewarded', '0.25 Rewarded', '0.5 Rewarded'
            ]
        else:  # choice
            contrast_conditions = [
                '0 Choice', '0.0625 Choice', '0.125 Choice', '0.25 Choice', '0.5 Choice'
            ]
    colormap = 'viridis'
    colors = sns.color_palette(colormap, len(contrast_conditions))[::-1]

    # Dictionary to save the traces of each session by condition
    traces_by_contrast = {cond: [] for cond in contrast_conditions}
    time_axis = None

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            session_name = recordingList.sessionName.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            try:
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                # Select the correct dictionary
                if alignment == 'reward':
                    dffTrace_mean = dffTrace_mean_reward
                elif alignment in ['stimulus', 'stim']:
                    dffTrace_mean = dffTrace_mean_stimuli
                elif alignment == 'choice':
                    dffTrace_mean = dffTrace_mean_choice
                else:
                    print(f"Unknown alignment: {alignment}")
                    continue
                # Time axis
                fRate = 30
                pre_stim_sec = 2
                total_time = 8
                n_frames = int(total_time * fRate)
                time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
                # Store traces for each condition
                for i, cond in enumerate(contrast_conditions):
                    data = dffTrace_mean.get(cond)
                    if data is not None and np.size(data) > 0:
                        mean_trace = np.nanmean(data, axis=0)
                        # Baseline subtraction
                        if baseline_window is not None:
                            baseline_start_idx = np.argmin(np.abs(time_axis - baseline_window[0]))
                            baseline_end_idx = np.argmin(np.abs(time_axis - baseline_window[1]))
                            if baseline_end_idx == baseline_start_idx:
                                baseline_end_idx += 1
                            baseline = np.nanmean(mean_trace[baseline_start_idx:baseline_end_idx])
                        else:
                            baseline_frames = int(0.2 * fRate)
                            baseline = np.nanmean(mean_trace[fRate*2 - baseline_frames:fRate*2])
                        mean_trace_bs = mean_trace - baseline
                        traces_by_contrast[cond].append(mean_trace_bs)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(contrast_conditions):
        traces = np.array(traces_by_contrast[cond])
        if traces.shape[0] > 0:
            mean_across_sessions = np.nanmean(traces, axis=0)
            sem_across_sessions = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
            ax.plot(time_axis[:len(mean_across_sessions)], mean_across_sessions, color=colors[i], label=cond, linewidth=2)
            ax.fill_between(
                time_axis[:len(mean_across_sessions)],
                mean_across_sessions - sem_across_sessions,
                mean_across_sessions + sem_across_sessions,
                color=colors[i], alpha=0.3
            )
    ax.axvline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel(f'Time from {alignment.capitalize()} (s)')
    ax.set_ylabel('df/f')
    if title is None:
        title = f'Mean ± SEM across sessions by contrast ({alignment}-aligned)'
    fig.suptitle(title, y=1.04, fontsize=14)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'mean_sem_across_sessions_{alignment}_by_contrasts', save_path)
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_ipsi_contra_vs_bias_correlation(
    recordingList,
    bias_df,
    time_window=[0.1, 0.8],
    bias_type='Choice Bias',
    alignment='choice',
    subfolder='responsive_neurons',
    save_path=None
):
    """
    Plots the correlation between the bias (performance difference at 0 contrast) and the ipsi-contra difference in a time window.
    Args:
        recordingList: DataFrame with session info
        bias_df: DataFrame with columns ['sessionName', 'diff_0_vs_chance']
        time_window: list, time window for mean df/f
        bias_type: 'Choice Bias' or 'Stim Bias'
        alignment: event alignment
        subfolder: subfolder for data
        save_path: where to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from scipy.stats import pearsonr

    # Time axis (assuming -2 to 6 s, 30 Hz)
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
    start_idx = int((time_window[0] + pre_stim_sec) * fRate)
    end_idx = int((time_window[1] + pre_stim_sec) * fRate)

    # Determine the key pair according to bias_type
    if bias_type.lower().startswith('choice'):
        key_ipsi = 'Choice Bias Ipsi'
        key_contra = 'Choice Bias Contra'
    elif bias_type.lower().startswith('stim'):
        key_ipsi = 'Stim Bias Ipsi'
        key_contra = 'Stim Bias Contra'
    else:
        print("bias_type must be 'Choice Bias' or 'Stim Bias'")
        return

    # List to store results
    results = []
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
            # Select the dictionary according to alignment
            if alignment.lower() == 'reward':
                dffTrace_mean = dffTrace_mean_reward
            elif alignment.lower() == 'stimulus':
                dffTrace_mean = dffTrace_mean_stimuli
            elif alignment.lower() == 'choice':
                dffTrace_mean = dffTrace_mean_choice
            else:
                print(f"alignment must be 'reward', 'stimulus' or 'choice' (in session {session_name})")
                continue
            data_ipsi = dffTrace_mean.get(key_ipsi)
            data_contra = dffTrace_mean.get(key_contra)
            if data_ipsi is not None and data_contra is not None:
                mean_ipsi = np.nanmean(data_ipsi, axis=0)
                mean_contra = np.nanmean(data_contra, axis=0)
                window_mean_ipsi = np.nanmean(mean_ipsi[start_idx:end_idx])
                window_mean_contra = np.nanmean(mean_contra[start_idx:end_idx])
                diff = window_mean_ipsi - window_mean_contra
                # Buscar bias en el DataFrame
                bias_row = bias_df[bias_df['sessionName'] == session_name]
                if not bias_row.empty:
                    bias_val = bias_row['diff_0_vs_chance'].values[0]
                    results.append({'sessionName': session_name, 'ipsi_contra_diff': diff, 'bias': bias_val})

    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    # Graficar correlación (eje x: bias, eje y: ipsi-contra)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(results_df['bias'], results_df['ipsi_contra_diff'], color='royalblue', s=60, edgecolor='k')
    # Ajuste lineal
    if len(results_df) > 1:
        m, b = np.polyfit(results_df['bias'], results_df['ipsi_contra_diff'], 1)
        ax.plot(results_df['bias'], m*results_df['bias']+b, color='orange', lw=2, label='Linear fit')
        r, p = pearsonr(results_df['bias'], results_df['ipsi_contra_diff'])
        ax.set_title(f'{bias_type} ({alignment}-aligned)\nCorrelation: r={r:.2f}, p={p:.3g}')
    else:
        ax.set_title(f'{bias_type} ({alignment}-aligned)\nCorrelation: N/A')
    ax.set_xlabel('Bias (0.5 - mean(0 contrast))')
    ax.set_ylabel('Ipsi - Contra (mean df/f in window)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_figureAll(f'{alignment}_aligned_by_{bias_type.replace(" ", "")}_ipsi_contra_correlation', save_path)
        plt.close(fig)
    else:
        plt.show()
    return results_df


def plot_event_aligned_stim_bias_diff_by_contrast_scatter_window_across_sessions(
    recordingList,
    time_window=[0.1, 0.8],
    alignment='stimulus',
    subfolder='responsive_neurons',
    save_path=None
):
    """
    For each session and contrast, calculate the mean in the time window and plot the ipsi-contra difference for each contrast (only Stim Bias).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from scipy.stats import f_oneway, ttest_ind

    # Time axis
    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)
    start_idx = int((time_window[0] + pre_stim_sec) * fRate)
    end_idx = int((time_window[1] + pre_stim_sec) * fRate)

    # Contrasts and keys
    contrasts = ['-0.0625', '-0.125', '-0.25', '-0.5', '0', '0.0625', '0.125', '0.25', '0.5']
    contrast_labels = ['-0.0625', '-0.125', '-0.25', '-0.5', '0', '0.0625', '0.125', '0.25', '0.5']
    if alignment.lower() == 'reward':
        key_ipsi = {c: f'{c} Rewarded Bias Ipsi' for c in contrasts}
        key_contra = {c: f'{c} Rewarded Bias Contra' for c in contrasts}
    elif alignment.lower() in ['stimulus', 'stim']:
        key_ipsi = {c: f'{c} Rewarded Bias Ipsi' for c in contrasts}
        key_contra = {c: f'{c} Rewarded Bias Contra' for c in contrasts}
    else:  # choice
        key_ipsi = {c: f'{c} Rewarded Bias Ipsi' for c in contrasts}
        key_contra = {c: f'{c} Rewarded Bias Contra' for c in contrasts}

    # Accumulate results
    window_means_by_contrast = {c: [] for c in contrasts}
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as f:
                dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
            if alignment.lower() == 'reward':
                dffTrace_mean = dffTrace_mean_reward
            elif alignment.lower() == 'stimulus':
                dffTrace_mean = dffTrace_mean_stimuli
            elif alignment.lower() == 'choice':
                dffTrace_mean = dffTrace_mean_choice
            else:
                print(f"alignment must be 'reward', 'stimulus' or 'choice' (in session {session_name})")
                continue
            for c in contrasts:
                data_ipsi = dffTrace_mean.get(key_ipsi[c])
                data_contra = dffTrace_mean.get(key_contra[c])
                if data_ipsi is not None and data_contra is not None:
                    mean_ipsi = np.nanmean(data_ipsi, axis=0)
                    mean_contra = np.nanmean(data_contra, axis=0)
                    window_mean_ipsi = np.nanmean(mean_ipsi[start_idx:end_idx])
                    window_mean_contra = np.nanmean(mean_contra[start_idx:end_idx])
                    diff = window_mean_ipsi - window_mean_contra
                    window_means_by_contrast[c].append(diff)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xvals = np.arange(len(contrasts))
    colors = plt.cm.viridis(np.linspace(0, 1, len(contrasts)))
    all_means, all_sems, all_vals = [], [], []
    for i, c in enumerate(contrasts):
        vals = window_means_by_contrast[c]
        if len(vals) > 0:
            ax.scatter([xvals[i]]*len(vals), vals, color=colors[i], alpha=0.7, s=60, edgecolor='k', label=f'{contrast_labels[i]} (n={len(vals)})')
            mean = np.nanmean(vals)
            sem = np.nanstd(vals)/np.sqrt(len(vals))
            all_means.append(mean)
            all_sems.append(sem)
            all_vals.append(vals)
        else:
            all_means.append(np.nan)
            all_sems.append(np.nan)
            all_vals.append([])
    ax.errorbar(xvals, all_means, yerr=all_sems, fmt='o', color='k', capsize=8, markersize=12, label='Mean ± SEM')
    ax.set_xticks(xvals)
    ax.set_xticklabels(contrast_labels)
    ax.set_xlim(-0.5, len(contrasts)-0.5)
    ax.set_ylabel(f'Ipsi - Contra df/f [{time_window[0]}, {time_window[1]}] s')
    ax.set_xlabel('Contrast')
    # Statistics
    valid_vals = [vals for vals in all_vals if len(vals) > 1]
    if len(valid_vals) >= 2:
        f_stat, p_val = f_oneway(*valid_vals)
        print(f'One-way ANOVA across contrasts: F = {f_stat:.3f}, p = {p_val:.4g}')
        pval_str = f'ANOVA p = {p_val:.3g}'
    else:
        pval_str = 'ANOVA p = N/A'
    print('\nPairwise t-tests (adjacent contrasts):')
    for i in range(len(contrasts)-1):
        if len(all_vals[i]) > 1 and len(all_vals[i+1]) > 1:
            tstat, pval = ttest_ind(all_vals[i], all_vals[i+1], nan_policy='omit')
            print(f'{contrast_labels[i]} vs {contrast_labels[i+1]}: t = {tstat:.3f}, p = {pval:.4g}')
    ax.set_title(f'{alignment.capitalize()}-aligned: Stim Bias by contrast (ipsi-contra diff)\n{pval_str}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'{alignment}_aligned_StimBias_by_contrast_ipsi_contra_diff_scatter_window_{time_window[0]}_{time_window[1]}', save_path)
        plt.close(fig)
    else:
        plt.show()
    return window_means_by_contrast

def plot_mean_sem_by_choice_across_sessions(
    recordingList,
    subfolder='responsive_neurons',
    alignment='choice',
    time_window=[-2, 6],
    save_path=None,
    title=None,
    key_pair=('Left Choices', 'Right Choices')
):
    """
    Plots the mean and SEM of neuronal activity aligned to choice,
    for a given pair of keys (e.g., ('Left Choices','Right Choices'), ('Choice Hemi Ipsi','Choice Hemi Contra'), etc.),
    averaging across all sessions.
    Args:
        recordingList: DataFrame with session info
        subfolder: subfolder where to look for the data
        alignment: 'choice' (default)
        time_window: time window to display (default [-2, 6] s)
        save_path: folder to save the figure (optional)
        title: figure title (optional)
        key_pair: tuple with the pair of keys to plot (default ('Left Choices','Right Choices'))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os

    # Time parameters
    fRate = 30
    pre_event_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_event_sec, total_time - pre_event_sec, n_frames)
    # Indices to crop the window
    start_idx = int((time_window[0] + pre_event_sec) * fRate)
    end_idx = int((time_window[1] + pre_event_sec) * fRate)

    key1, key2 = key_pair
    traces_1 = []
    traces_2 = []

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_path = recordingList.analysispathname[ind]
            session_name = recordingList.sessionName[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            try:
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                dffTrace_mean = dffTrace_mean_choice
                data_1 = dffTrace_mean.get(key1)
                data_2 = dffTrace_mean.get(key2)
                if data_1 is not None and np.size(data_1) > 0:
                    mean_trace = np.nanmean(data_1, axis=0)
                    traces_1.append(mean_trace[start_idx:end_idx])
                if data_2 is not None and np.size(data_2) > 0:
                    mean_trace = np.nanmean(data_2, axis=0)
                    traces_2.append(mean_trace[start_idx:end_idx])
            except Exception:
                continue
    traces_1 = np.array(traces_1)
    traces_2 = np.array(traces_2)
    t = time_axis[start_idx:end_idx]

    # Calculate mean and SEM
    if traces_1.shape[0] > 0:
        mean_1 = np.nanmean(traces_1, axis=0)
        sem_1 = np.nanstd(traces_1, axis=0) / np.sqrt(traces_1.shape[0])
    else:
        mean_1 = np.array([])
        sem_1 = np.array([])
    if traces_2.shape[0] > 0:
        mean_2 = np.nanmean(traces_2, axis=0)
        sem_2 = np.nanstd(traces_2, axis=0) / np.sqrt(traces_2.shape[0])
    else:
        mean_2 = np.array([])
        sem_2 = np.array([])

    # Set colors: key1 in blue, key2 in red
    color1 = 'royalblue'
    color2 = 'crimson'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(mean_1) > 0 and len(mean_1) == len(t):
        ax.plot(t, mean_1, color=color1, label=f'{key1} (n={traces_1.shape[0]})', linewidth=2)
        ax.fill_between(t, mean_1 - sem_1, mean_1 + sem_1, color=color1, alpha=0.3)
    if len(mean_2) > 0 and len(mean_2) == len(t):
        ax.plot(t, mean_2, color=color2, label=f'{key2} (n={traces_2.shape[0]})', linewidth=2)
        ax.fill_between(t, mean_2 - sem_2, mean_2 + sem_2, color=color2, alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Time from Choice (s)')
    ax.set_ylabel('df/f')
    if title is None:
        title = f'Mean ± SEM across sessions: {key1} vs {key2} (choice-aligned)'
    ax.set_title(title)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'mean_sem_across_sessions_{key1}_vs_{key2}', save_path)
        plt.close(fig)
    else:
        plt.show()

def scatter_peak_window_mean_by_key_pair(
    recordingList,
    subfolder='responsive_neurons',
    alignment='choice',
    window_ms=500,
    save_path=None,
    title=None,
    key_pair=('Left Choices', 'Right Choices'),
    colors=('royalblue', 'crimson')
):
    """
    For a given pair of keys, finds the peak of the global mean curve, takes a window of ±window_ms/2 ms around the peak,
    calculates the mean in that window for each session, plots the values in a scatter plot (custom colors),
    and performs a t-test between the two groups. Points are centered and jittered for better visualization.
    Args:
        recordingList: DataFrame with session info
        subfolder: subfolder where to look for the data
        alignment: 'choice' (default)
        window_ms: window size in ms (default 500)
        save_path: folder to save the figure (optional)
        title: figure title (optional)
        key_pair: tuple with the pair of keys to compare (default ('Left Choices','Right Choices'))
        colors: tuple/list of two colors for the two groups (default ('royalblue', 'crimson'))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from scipy.stats import ttest_ind

    # Time parameters
    fRate = 30  # Hz
    pre_event_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_event_sec, total_time - pre_event_sec, n_frames)
    window_frames = int((window_ms/1000) * fRate / 2)  # half window in frames

    key1, key2 = key_pair
    color1, color2 = colors
    traces_1 = []
    traces_2 = []

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            session_path = recordingList.analysispathname[ind]
            session_name = recordingList.sessionName[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pickle_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pickle_file):
                continue
            try:
                with open(pickle_file, 'rb') as f:
                    dffTrace_mean_reward, dffTrace_mean_stimuli, dffTrace_mean_choice = pickle.load(f)
                dffTrace_mean = dffTrace_mean_choice
                data_1 = dffTrace_mean.get(key1)
                data_2 = dffTrace_mean.get(key2)
                if data_1 is not None and np.size(data_1) > 0:
                    mean_trace = np.nanmean(data_1, axis=0)
                    traces_1.append(mean_trace)
                if data_2 is not None and np.size(data_2) > 0:
                    mean_trace = np.nanmean(data_2, axis=0)
                    traces_2.append(mean_trace)
            except Exception:
                continue
    traces_1 = np.array(traces_1)
    traces_2 = np.array(traces_2)

    # Global mean
    global_mean_1 = np.nanmean(traces_1, axis=0) if traces_1.shape[0] > 0 else np.array([])
    global_mean_2 = np.nanmean(traces_2, axis=0) if traces_2.shape[0] > 0 else np.array([])

    # Find global peak (of both curves together)
    if len(global_mean_1) > 0 and len(global_mean_2) > 0:
        global_mean = (global_mean_1 + global_mean_2) / 2
        peak_idx = np.nanargmax(np.abs(global_mean))
    elif len(global_mean_1) > 0:
        peak_idx = np.nanargmax(np.abs(global_mean_1))
    elif len(global_mean_2) > 0:
        peak_idx = np.nanargmax(np.abs(global_mean_2))
    else:
        print("No data to plot.")
        return

    start_idx = max(0, peak_idx - window_frames)
    end_idx = min(n_frames, peak_idx + window_frames + 1)

    # Calculate mean in window for each session
    means_1 = [np.nanmean(trace[start_idx:end_idx]) for trace in traces_1]
    means_2 = [np.nanmean(trace[start_idx:end_idx]) for trace in traces_2]

    # Scatter plot positions and jitter
    center1 = 0.25
    center2 = 0.75
    jitter_scale = 0.06
    np.random.seed(0)  # For reproducibility
    x1 = center1 + np.random.uniform(-jitter_scale, jitter_scale, size=len(means_1))
    x2 = center2 + np.random.uniform(-jitter_scale, jitter_scale, size=len(means_2))

    # Plot scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x1, means_1, color=color1, label=f'{key1} (n={len(means_1)})', s=80, alpha=0.8)
    ax.scatter(x2, means_2, color=color2, label=f'{key2} (n={len(means_2)})', s=80, alpha=0.8)
    # Mean and SEM
    for x, vals, color in zip([center1, center2], [means_1, means_2], [color1, color2]):
        if len(vals) > 0:
            mean = np.nanmean(vals)
            sem = np.nanstd(vals) / np.sqrt(len(vals))
            ax.errorbar(x, mean, yerr=sem, fmt='o', color='k', capsize=8, markersize=18, alpha=0.8)
    # Statistics
    if len(means_1) > 1 and len(means_2) > 1:
        tstat, pval = ttest_ind(means_1, means_2, nan_policy='omit')
        pval_str = f't-test p = {pval:.3g}'
    else:
        pval_str = 't-test p = N/A'
    # Labels
    ax.set_xticks([center1, center2])
    ax.set_xticklabels([key1, key2], rotation=20)
    ax.set_ylabel(f'Mean df/f [{time_axis[start_idx]:.2f}, {time_axis[end_idx-1]:.2f}] s')
    # Ajustar límites del eje x para dejar margen
    ax.set_xlim(center1 - 3*jitter_scale, center2 + 3*jitter_scale)
    if title is None:
        title = f'Mean df/f in {window_ms} ms window around global peak\n{key1} vs {key2}\n{pval_str}'
    else:
        title = f'{title}\n{pval_str}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll(f'scatter_peakwindow_mean_{key1}_vs_{key2}', save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_reward_aligned_rewarded_vs_unrewarded_across_sessions(
    recordingList,
    subfolder='responsive_neurons',
    save_path=None
):
    """
    Plots the mean and SEM across sessions of the traces for Rewarded and Unrewarded trials,
    aligned to reward. Rewarded in red, Unrewarded in black.
    Args:
        recordingList: DataFrame with session info
        subfolder: subfolder where to look for the data
        save_path: folder to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os

    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    traces_rewarded = []
    traces_unrewarded = []

    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted.iloc[ind] == 1:
            session_path = recordingList.analysispathname.iloc[ind]
            subfolder_path = os.path.join(session_path, subfolder)
            pkl_file = os.path.join(subfolder_path, 'imaging-dffTrace_mean.pkl')
            if not os.path.exists(pkl_file):
                continue
            try:
                with open(pkl_file, 'rb') as f:
                    dffTrace_mean_reward, _, _ = pickle.load(f)
                trace_rewarded = dffTrace_mean_reward.get('Rewarded')
                trace_unrewarded = dffTrace_mean_reward.get('Unrewarded')
                if trace_rewarded is not None and np.size(trace_rewarded) > 0:
                    mean_trace = np.nanmean(trace_rewarded, axis=0)
                    traces_rewarded.append(mean_trace)
                if trace_unrewarded is not None and np.size(trace_unrewarded) > 0:
                    mean_trace = np.nanmean(trace_unrewarded, axis=0)
                    traces_unrewarded.append(mean_trace)
            except Exception as e:
                print(f"Error processing session {getattr(recordingList, 'sessionName', [''])[ind]}: {str(e)}")
                continue
    traces_rewarded = np.array(traces_rewarded)
    traces_unrewarded = np.array(traces_unrewarded)

    # Calcular media y SEM
    if traces_rewarded.shape[0] > 0:
        mean_rewarded = np.nanmean(traces_rewarded, axis=0)
        sem_rewarded = np.nanstd(traces_rewarded, axis=0) / np.sqrt(traces_rewarded.shape[0])
    else:
        mean_rewarded = np.array([])
        sem_rewarded = np.array([])
    if traces_unrewarded.shape[0] > 0:
        mean_unrewarded = np.nanmean(traces_unrewarded, axis=0)
        sem_unrewarded = np.nanstd(traces_unrewarded, axis=0) / np.sqrt(traces_unrewarded.shape[0])
    else:
        mean_unrewarded = np.array([])
        sem_unrewarded = np.array([])

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(mean_rewarded) > 0:
        ax.plot(time_axis[:len(mean_rewarded)], mean_rewarded, color='crimson', label=f'Rewarded (n={traces_rewarded.shape[0]})', linewidth=2)
        ax.fill_between(time_axis[:len(mean_rewarded)], mean_rewarded - sem_rewarded, mean_rewarded + sem_rewarded, color='crimson', alpha=0.3)
    if len(mean_unrewarded) > 0:
        ax.plot(time_axis[:len(mean_unrewarded)], mean_unrewarded, color='black', label=f'Unrewarded (n={traces_unrewarded.shape[0]})', linewidth=2)
        ax.fill_between(time_axis[:len(mean_unrewarded)], mean_unrewarded - sem_unrewarded, mean_unrewarded + sem_unrewarded, color='black', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('df/f')
    ax.set_title('Reward-aligned: Rewarded vs Unrewarded (mean ± SEM across sessions)')
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path is not None:
        save_figureAll('reward_aligned_rewarded_vs_unrewarded_across_sessions', save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_mean_dff_by_contrast_hemi_bias_4panels(
    recordingList,
    event_type='stimulus',
    time_window=[0.1, 0.8],
    subfolder='responsive_neurons',
    save_path=None,
    keys_hemi_ipsi_bias_ipsi=None,
    keys_hemi_ipsi_bias_contra=None,
    keys_hemi_contra_bias_ipsi=None,
    keys_hemi_contra_bias_contra=None,
    use_zscored=True,
    title=None
):
    """
    Plot the mean df/f for each contrast, collapsed into 5 absolute values (0, 0.0625, 0.125, 0.25, 0.5),
    for 4 groups (Hemi Ipsi - Bias Ipsi, Hemi Ipsi - Bias Contra, Hemi Contra - Bias Ipsi, Hemi Contra - Bias Contra),
    each in its own subplot (2x2).
    By default, uses the keys provided by the user.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pickle
    import os

    abs_contrast_values = [0, 0.0625, 0.125, 0.25, 0.5]
    abs_contrast_labels = [str(c) for c in abs_contrast_values]
    contrast_colors = sns.color_palette('viridis', len(abs_contrast_values))[::-1]

    # Default keys from user
    if keys_hemi_ipsi_bias_ipsi is None:
        keys_hemi_ipsi_bias_ipsi = [
            '-0.0625 Rewarded Hemi Ipsi Bias Ipsi','-0.125 Rewarded Hemi Ipsi Bias Ipsi', '-0.25 Rewarded Hemi Ipsi Bias Ipsi', '-0.5 Rewarded Hemi Ipsi Bias Ipsi',
            '0.0625 Rewarded Hemi Ipsi Bias Ipsi','0.125 Rewarded Hemi Ipsi Bias Ipsi', '0.25 Rewarded Hemi Ipsi Bias Ipsi', '0.5 Rewarded Hemi Ipsi Bias Ipsi', '0 Rewarded Hemi Ipsi Bias Ipsi'
        ]
    if keys_hemi_ipsi_bias_contra is None:
        keys_hemi_ipsi_bias_contra = [
            '-0.0625 Rewarded Hemi Ipsi Bias Contra','-0.125 Rewarded Hemi Ipsi Bias Contra', '-0.25 Rewarded Hemi Ipsi Bias Contra', '-0.5 Rewarded Hemi Ipsi Bias Contra',
            '0.0625 Rewarded Hemi Ipsi Bias Contra','0.125 Rewarded Hemi Ipsi Bias Contra', '0.25 Rewarded Hemi Ipsi Bias Contra', '0.5 Rewarded Hemi Ipsi Bias Contra', '0 Rewarded Hemi Ipsi Bias Contra'
        ]
    if keys_hemi_contra_bias_ipsi is None:
        keys_hemi_contra_bias_ipsi = [
            '-0.0625 Rewarded Hemi Contra Bias Ipsi','-0.125 Rewarded Hemi Contra Bias Ipsi', '-0.25 Rewarded Hemi Contra Bias Ipsi', '-0.5 Rewarded Hemi Contra Bias Ipsi',
            '0.0625 Rewarded Hemi Contra Bias Ipsi','0.125 Rewarded Hemi Contra Bias Ipsi', '0.25 Rewarded Hemi Contra Bias Ipsi', '0.5 Rewarded Hemi Contra Bias Ipsi', '0 Rewarded Hemi Contra Bias Ipsi'
        ]
    if keys_hemi_contra_bias_contra is None:
        keys_hemi_contra_bias_contra = [
            '-0.0625 Rewarded Hemi Contra Bias Contra','-0.125 Rewarded Hemi Contra Bias Contra', '-0.25 Rewarded Hemi Contra Bias Contra', '-0.5 Rewarded Hemi Contra Bias Contra',
            '0.0625 Rewarded Hemi Contra Bias Contra','0.125 Rewarded Hemi Contra Bias Contra', '0.25 Rewarded Hemi Contra Bias Contra', '0.5 Rewarded Hemi Contra Bias Contra', '0 Rewarded Hemi Contra Bias Contra'
        ]

    group_keys = [
        (keys_hemi_ipsi_bias_ipsi, 'Hemi Ipsi - Bias Ipsi'),
        (keys_hemi_ipsi_bias_contra, 'Hemi Ipsi - Bias Contra'),
        (keys_hemi_contra_bias_ipsi, 'Hemi Contra - Bias Ipsi'),
        (keys_hemi_contra_bias_contra, 'Hemi Contra - Bias Contra')
    ]

    def get_abs_contrast(cond):
        for val in abs_contrast_values:
            if cond.startswith(f'{val} ') or cond.startswith(f'-{val} '):
                return val
        return None

    # Prepare data structure
    abs_contrast_data = {name: {v: [] for v in abs_contrast_values} for _, name in group_keys}
    session_points = {name: {} for _, name in group_keys}

    # Process each session
    for ind, recordingDate in enumerate(recordingList.recordingDate):
        if recordingList.imagingDataExtracted[ind] == 1:
            animal_id = recordingList.animalID[ind]
            session_name = recordingList.sessionName[ind]
            pathname = recordingList.analysispathname[ind]
            subfolder_path = os.path.join(pathname, subfolder)
            try:
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
                for keys, group_name in group_keys:
                    session_contrasts = []
                    session_means = []
                    for cond in keys:
                        abs_val = get_abs_contrast(cond)
                        if abs_val is not None and cond in dffTrace_mean and dffTrace_mean[cond] is not None:
                            mean_dff = np.nanmean(dffTrace_mean[cond])
                            abs_contrast_data[group_name][abs_val].append({'animal': animal_id, 'session': session_name, 'mean_dff': mean_dff, 'contrast': abs_val})
                            session_contrasts.append(abs_val)
                            session_means.append(mean_dff)
                    if len(session_contrasts) > 1:
                        session_points[group_name][session_name] = (session_contrasts, session_means)
            except Exception as e:
                print(f"Error processing session {session_name}: {str(e)}")
                continue

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for idx, (keys, group_name) in enumerate(group_keys):
        ax = axes[idx]
        # Connect points of each session
        for session_name, (xvals, yvals) in session_points[group_name].items():
            ax.plot(xvals, yvals, linestyle='--', color='gray', alpha=0.5, linewidth=1, zorder=1)
        # Individual points per absolute contrast
        for i, abs_val in enumerate(abs_contrast_values):
            if abs_contrast_data[group_name][abs_val]:
                df_contrast = pd.DataFrame(abs_contrast_data[group_name][abs_val])
                ax.scatter(df_contrast['contrast'], df_contrast['mean_dff'], 
                          color=contrast_colors[i], alpha=0.7, s=50, 
                          label=f'Contrast {abs_contrast_labels[i]}', zorder=2)
        # Global mean and SEM
        means = []
        sems = []
        for abs_val in abs_contrast_values:
            vals = [d['mean_dff'] for d in abs_contrast_data[group_name][abs_val]]
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(np.std(vals)/np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(np.nan)
        for i, (x, m, s) in enumerate(zip(abs_contrast_values, means, sems)):
            if not np.isnan(m):
                ax.errorbar(x, m, yerr=s, fmt='_', color='k', elinewidth=3, capsize=7, alpha=0.8, markersize=18, zorder=4)
        ax.set_xlabel('Contrast (absolute value)')
        ax.set_title(group_name)
        ax.set_xscale('linear')
        ax.set_xticks(abs_contrast_values)
        ax.set_xticklabels(abs_contrast_labels)
        import matplotlib.pyplot as plt
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_ylabel(f'Mean df/f')
    axes[2].set_ylabel(f'Mean df/f')
    plt.tight_layout()
    if title is None:
        title = f'Mean df/f by absolute contrast ({time_window[0]}-{time_window[1]}s post-{event_type}) - 4 groups (Hemi/Bias) - {subfolder} (baseline subtracted)'
    fig.suptitle(title, y=1.02, fontsize=15)
    if save_path is not None:
        save_figureAll(f'mean_dff_by_contrast_hemi_bias_4panels_{event_type}_{time_window[0]}_{time_window[1]}s_{subfolder}', save_path)
        plt.close(fig)
    else:
        plt.show()
    return fig