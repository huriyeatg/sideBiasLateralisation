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

def lineplot_sessions(dffTrace_mean, analysis_params, colormap,
                    duration, zscoreRun, savefigname, savefigpath, baseline_subtract=None, title=None):
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
        fig, axes = plt.subplots(nrows=1, ncols=len(analysis_params)+1, figsize=((len(analysis_params)+1)*2, 6), 
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

def plot_combined_psychometric(info, save_path=None):
    """
    Creates two plots:
    1. Individual right turn probability vs contrast for each session
    2. Combined mean and standard error across all sessions
    
    Args:
        info: Info object containing recordingList with session information
        save_path: Path where to save the figure. If None, uses default path
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

    # Set default save path if none provided
    if save_path is None:
        save_path = r"C:\Users\Lak Lab\Documents\Github\sideBiasLateralisation\analysis\figs"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get animal ID from first session
    animal_id = info.recordingList.animalID[0]
    
    # Create filename with animal ID
    full_save_path = os.path.join(save_path, f'{animal_id}_combined_psychometric.png')
    
    plt.savefig(full_save_path)
    plt.close()
    
    print(f"Figure saved at: {full_save_path}")



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
    plt.figure(figsize=(18, 5))
    
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
    plt.figure(figsize=(15, 6))
    
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
                                              exclude_zero_contrast=True, figsize=(12, 8)):
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
                                           figsize=(15, 60), n_cols=3, time_window=[-2, 6], baseline_window=None,
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
        contrast_conditions = ['0.5 reward', '0.25 reward', '0.125 reward', '0.0625 reward', '0 reward']
    if contrast_labels is None:
        contrast_labels = ['0.5', '0.25', '0.125', '0.0625', '0']
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
                                                   figsize=(15, 60), n_cols=3, time_window=[-2, 6], baseline_window=None,
                                                   contrast_conditions=None, contrast_labels=None):
    """
    Plots individual cell traces (z-scored) aligned to stimulus, separated by contrast with viridis colormap.
    Permite sustracción de baseline.
    Permite especificar condiciones de contraste y etiquetas.
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
    Plots and saves one figure per neuron, showing all available contrast traces for that neuron.
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
        plt.xlabel('Time (s)')
        plt.ylabel('dF/F')
        plt.title(f'{session_name} - Cell {cell_idx+1}')
        plt.xlim(time_window)
        plt.legend(title='Contrast', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save figure
        if single_neuron_dir is not None:
            fname = os.path.join(single_neuron_dir, f'{session_name}_cell{cell_idx+1}.png')
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    print("Done plotting all single-neuron traces.")