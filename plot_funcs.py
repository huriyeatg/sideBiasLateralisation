# This code  has plotting functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import statsmodels.api as sm
from scipy import stats


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

def lineplot_sessions(dffTrace_mean,analysis_params, colormap,
                    duration,zscoreRun, savefigname, savefigpath ) :
    
    color =  sns.color_palette(colormap, len(analysis_params))
    sessionsData ={}

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]   
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]    
            array = np.reshape(array, (nCell, analysis_window))
            if zscoreRun:
                sessionsData[indx]= zscore(array, axis = 1)
            else:
                sessionsData[indx]= array
    step = 30 # for x ticks
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        yaxis_length = int(duration[0])*30
    plt.subplot(2,2, 1)
    for idx, sessionData in enumerate(sessionsData):
        plot_data = sessionsData[idx]
        if type(plot_data) != type(None):
            
            x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype = int)
            xticks = np.arange(0, len(x_labels), step)
            xticklabels = x_labels[::step]
            df = pd.DataFrame(plot_data).melt()
            # Smooth the data using lowess method from statsmodels
            x=df['variable']
            y=df['value']
            lowess_smoothed = sm.nonparametric.lowess(y, x,frac=0.1)
            ax = sns.lineplot(x=x, y=y, data=df, color=color[idx], 
                              label=analysis_params[idx])
            ax = sns.lineplot(x=lowess_smoothed[:, 0], y=lowess_smoothed[:, 1], 
                              data=df, color=color[idx], linewidth = 3 )

            plt.axvline(x=60, color='k', linestyle='--')
            plt.xticks (ticks = xticks, labels= xticklabels)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.xlim(30,60+yaxis_length)
            plt.xlabel('Time (sec)')
            if zscoreRun:
                plt.ylabel('DFF(zscore)')
            else:
                plt.ylabel('DFF')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            #plt.title(analysis_params[idx])
    save_figureAll(savefigname,savefigpath)

def heatmap_sessions(dffTrace_mean,analysis_params, colormap,
                       selectedSession, duration, savefigname, savefigpath ) :
    ## Parameters
    fRate = 1000/30
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    analysisWindowDur = 500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))

    sessionsData ={}

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
        fig, axes = plt.subplots(nrows=1, ncols=len(analysis_params)+1, figsize=((len(analysis_params)+1)*2, 12), 
                                gridspec_kw={'width_ratios': grid_ratio})

        for idx, sessionData in enumerate(sessionsData):
            plot_data = sessionsData[idx]
            if type(plot_data) != type(None):
                if selectedSession == 'WithinSession':
                    sortedInd = np.array(np.nanmean(plot_data[:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]
                else:
                    sortedInd = np.array(np.nanmean(sessionsData[selectedSession][:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]

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
        fig, axs = plt.subplots((len(sessionsData)+1), 1, figsize=(6, (6*(len(sessionsData)+1))))
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

def plot_combined_psychometric(info):
    """
    Creates two plots:
    1. Individual right turn probability vs contrast for each session
    2. Combined mean and standard error across all sessions
    
    Args:
        info: Info object containing recordingList with session information
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import pandas as pd
    import os
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dictionary to store probabilities for each contrast across sessions
    contrast_probs = {}
    
    # For each session in recordingList
    for ind in range(len(info.recordingList)):
        try:
            session = info.recordingList.sessionName[ind]
            print(f"\nProcessing session: {session}")
            
            # Construct CSV path
            csv_path = os.path.join(info.recordingList.analysispathname[ind], 
                                  f"{session}_CorrectedeventTimes.csv")
            #print(f"Looking for CSV at: {csv_path}")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"CSV file not found at: {csv_path}")
                continue
                
            # Read CSV file
            #print(f"Reading CSV from: {csv_path}")
            b = pd.read_csv(csv_path)
            
            # Filter good trials
            good_trials = (b['repeatNumber'] == 1) & (b['choice'] != 'NoGo')
            
            # Calculate contrast difference
            c_diff = b['contrastRight'] - b['contrastLeft']
            
            # Calculate right turn probability for each contrast
            unique_contrasts = np.unique(c_diff)
            right_probs = []
            right_probs_ci = []
            
            for contrast in unique_contrasts:
                trials = (c_diff == contrast) & good_trials
                if np.sum(trials) > 0:
                    right_choices = (b['choice'][trials] == 'Right')
                    prob = np.mean(right_choices)
                    # Calculate binomial confidence interval
                    ci = stats.binom.interval(0.95, np.sum(trials), prob)
                    right_probs.append(prob)
                    right_probs_ci.append([prob - ci[0]/np.sum(trials), 
                                         ci[1]/np.sum(trials) - prob])
                    
                    # Store probability for combined plot
                    if contrast not in contrast_probs:
                        contrast_probs[contrast] = []
                    contrast_probs[contrast].append(prob)
            
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


def plot_combined_response_time(info, analysis_path):
    """
    Creates four plots:
    1. Individual response time vs contrast for each session
    2. Combined mean and standard error across all sessions
    3. Individual response time vs contrast for each session, separated by ipsi/contra
    4. Combined mean and standard error across all sessions, separated by ipsi/contra
    
    Args:
        info: Info object containing recordingList with session information
        analysis_path: String with the path to the analysis folder
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
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
    plt.show()

  
def update_bias_data(new_id, new_bias):
    """
    Updates the JSON file with new IDs and their biases.
    
    Args:
        new_id: String with the new ID (e.g., "MBL015")
        new_bias: String with the bias ("Left" or "Right")
    """

    import json
    import os
    # JSON file path
    analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
    json_path = os.path.join(analysis_path, 'bias_data.json')
    
    # Load existing data if file exists
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    # Update with new ID and bias
    existing_data[new_id] = new_bias
    
    # Save updated data
    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
