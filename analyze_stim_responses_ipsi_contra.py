from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import numpy as np
from scipy.stats import wilcoxon, ranksums
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from matplotlib_venn import venn2
from iblatlas.atlas import AllenAtlas
from scipy.stats import binomtest        

one = ONE(base_url='https://openalyx.internationalbrainlab.org',cache_dir='//qnap-al001/Data/ibl_brain_wide_map')

# make a list of brain regions to analyze
brain_regions = ['VISp','MOs', 'MOp', 'SSp', 'TEa']
#brain_regions = ['PF']
# define time windows for psth
# stimulus window can be early [0,0.1] or late [0.1,0.3]. Specify both so that we can loop over them
stimulus_windows = [[0, 0.1], [0.1, 0.3]] # time window for stimulus
baseline_window = [-0.1, 0] # time window for baseline
time_bins = np.arange(-0.1, 0.3, 0.01)  # 10 ms bins for psth

tag = 'Brainwidemap'

ba = AllenAtlas()

# loop over brain regions
for brain_acronym in brain_regions:
    
    # loop over stimulus windows
    for stimulus_window in stimulus_windows:

        insertions = one.search_insertions(atlas_acronym=brain_acronym, tag=tag)
        print(f' No. of insertions: {len(insertions)}')

        psth_contra_all_clusters = np.empty((0, len(time_bins)))
        psth_ipsi_all_clusters = np.empty((0, len(time_bins)))

        # make a cluster_df_all dataframe to store the results for all clusters
        cluster_df_all = pd.DataFrame(columns=['cluster_id', 'eid', 'atlas_id', 'x', 'y', 'z', 'brain_area_label', 'mean_fr_contra', 'mean_fr_ipsi', 'p_contra', 'p_ipsi', 'p_contra_ipsi'])

        # for loop over insertions
        for i, insertion in enumerate(insertions):

            # print insertion number
            print(f'Insertion {i+1}/{len(insertions)}')

            pid = insertion
            eid, _ = one.pid2eid(pid)

            # load trial data
            obj = 'trials'
            trials = one.load_object(eid, obj)
            print(trials.keys())

            # compute response times by subtracting stimOn_times from response_times
            trials['response_times'] = trials['response_times'] - trials['stimOn_times']

            # get stimulus onset times for left and right high contrast stimuli, feedbackType must be 1 and response_times < 2
            left_stimOn_times = trials['stimOn_times'][(trials['contrastLeft'] == 1) & (trials['feedbackType'] == 1) & (trials['response_times'] < 2)]
            right_stimOn_times = trials['stimOn_times'][(trials['contrastRight'] == 1) & (trials['feedbackType'] == 1) & (trials['response_times'] < 2)]

            # load spike data
            spike_loader = SpikeSortingLoader(pid=pid, one=one)
            spikes, clusters, channels = spike_loader.load_spike_sorting()
            clusters = spike_loader.merge_clusters(spikes, clusters, channels)

            if brain_acronym == 'VISp':
                # Only include clusters with label == 1 and acronyms that equal 'VISp' or start with 'VISp' followed by a number.
                valid_acronym = [acronym == 'VISp' or (acronym.startswith('VISp') and len(acronym) > 4 and acronym[4].isdigit())
                                 for acronym in clusters['acronym'].astype(str)]
                clusters_idx = np.where(np.bitwise_and(clusters['label'] == 1, np.array(valid_acronym)))[0]
            # else if brain_acronym == 'PF', cluster acronyms should match PF exactly
            elif brain_acronym == 'PF':
                clusters_idx = np.where(np.bitwise_and(clusters['label'] == 1,
                                                     clusters['acronym'].astype(str) == brain_acronym))[0]
            else:
                clusters_idx = np.where(np.bitwise_and(clusters['label'] == 1,
                                                         np.char.startswith(clusters['acronym'].astype(str), brain_acronym)))[0]
            
            spikes_idx = np.isin(spikes.clusters, clusters_idx)

            clusters_roi = {k: v[clusters_idx] for k, v in clusters.items()}
            spikes_roi = {k: v[spikes_idx] for k, v in spikes.items()}

            for clust in clusters_roi['cluster_id']:
                
                # get spike times for this cluster
                spike_times = spikes_roi['times'][spikes_roi['clusters'] == clust]

                # if x coordinate is negative save left_stimOn_times as ipsi_stimOn_times and right_stimOn_times as contra_stimOn_times
                # if x coordinate is positive save left_stimOn_times as contra_stimOn_times and right_stimOn_times as ipsi_stimOn_times
                if clusters_roi['x'][clusters_roi['cluster_id'] == clust] < 0:
                    ipsi_stimOn_times = left_stimOn_times
                    contra_stimOn_times = right_stimOn_times
                else:
                    ipsi_stimOn_times = right_stimOn_times
                    contra_stimOn_times = left_stimOn_times


                ## confirm atlas location of the cluster
                mlapdv = np.c_[clusters_roi["x"][clusters_roi['cluster_id'] == clust], clusters_roi["y"][clusters_roi['cluster_id'] == clust], clusters_roi["z"][clusters_roi['cluster_id'] == clust]]
                area_label = ba.get_labels(mlapdv, mapping="Allen-lr")

                ## CONTRALATERAL STIMULI ##

                # initialize empty array for psth_contra
                psth_contra= np.empty((0, len(time_bins)-1))
                baseline_spikes_contra= np.empty((0, 1))
                stimulus_spikes_contra= np.empty((0, 1))

                for contra_stimulus in contra_stimOn_times:
                    
                    # PSTH
                    # get the psth for this cluster and contra stimulus
                    psth = np.histogram(spike_times, bins=time_bins+contra_stimulus)
                    # append psth to array for this cluster
                    psth_contra = np.vstack((psth_contra, psth[0]))

                    # get number of spikes in the baseline window
                    baseline_spikes = np.histogram(spike_times, bins=baseline_window+contra_stimulus)[0]
                    stimulus_spikes = np.histogram(spike_times, bins=stimulus_window+contra_stimulus)[0]
                    # append baseline_spikes to baseline_spikes_contra array
                    baseline_spikes_contra = np.vstack((baseline_spikes_contra, baseline_spikes))
                    # append stimulus_spikes to stimulus_spikes_contra array
                    stimulus_spikes_contra = np.vstack((stimulus_spikes_contra, stimulus_spikes))

                ## IPSILATERAL STIMULI ##

                # initialize empty array for psth_ipsi
                psth_ipsi = np.empty((0, len(time_bins)-1))
                baseline_spikes_ipsi = np.empty((0, 1))
                stimulus_spikes_ipsi = np.empty((0, 1))

                for ipsi_stimulus in ipsi_stimOn_times:
                    
                    # PSTH
                    # get the psth for this cluster and ipsi stimulus
                    psth = np.histogram(spike_times, bins=time_bins+ipsi_stimulus)
                    # append psth to array for this cluster
                    psth_ipsi = np.vstack((psth_ipsi, psth[0]))

                    # get number of spikes in the baseline window
                    baseline_spikes = np.histogram(spike_times, bins=baseline_window+ipsi_stimulus)[0]
                    stimulus_spikes = np.histogram(spike_times, bins=stimulus_window+ipsi_stimulus)[0]
                    # append baseline_spikes to baseline_spikes_ipsi array
                    baseline_spikes_ipsi = np.vstack((baseline_spikes_ipsi, baseline_spikes))
                    # append stimulus_spikes to stimulus_spikes_ipsi array
                    stimulus_spikes_ipsi = np.vstack((stimulus_spikes_ipsi, stimulus_spikes))


                # summary statistics for this cluster and statistical tests

                ## PSTH CONTRALATERAL ##
                # compute the mean of the psth_contra array along axis 0 (across trials)
                psth_contra = np.mean(psth_contra, axis=0)

                # make psth_contra a row vector
                psth_contra = np.reshape(psth_contra, (1, -1))
                # add the cluster id to the psth_contra array
                psth_contra = np.hstack((np.full((psth_contra.shape[0], 1), clust), psth_contra))
                # add psth_contra to the psth_contra_all_clusters array
                psth_contra_all_clusters = np.vstack((psth_contra_all_clusters, psth_contra))

                ## PSTH IPSILATERAL ##
                # compute the mean of the psth_ipsi array along axis 0 (across trials)
                psth_ipsi = np.mean(psth_ipsi, axis=0)
                # make psth_ipsi a row vector
                psth_ipsi = np.reshape(psth_ipsi, (1, -1))
                # add the cluster id to the psth_ipsi array
                psth_ipsi = np.hstack((np.full((psth_ipsi.shape[0], 1), clust), psth_ipsi))
                # add psth_ipsi to the psth_ipsi_all_clusters array
                psth_ipsi_all_clusters = np.vstack((psth_ipsi_all_clusters, psth_ipsi))

                ## Binned firing rate ##
                stimulus_firing_rate_contra = stimulus_spikes_contra / (stimulus_window[1] - stimulus_window[0])
                stimulus_firing_rate_ipsi = stimulus_spikes_ipsi / (stimulus_window[1] - stimulus_window[0])
                baseline_firing_rate_contra = baseline_spikes_contra / (baseline_window[1] - baseline_window[0])
                baseline_firing_rate_ipsi = baseline_spikes_ipsi / (baseline_window[1] - baseline_window[0])

                # non-parametric paired-test between baseline and stimulus window
                p_contra = wilcoxon(stimulus_firing_rate_contra,baseline_firing_rate_contra, alternative='greater').pvalue[0]
                p_ipsi = wilcoxon(stimulus_firing_rate_ipsi,baseline_firing_rate_ipsi, alternative='greater').pvalue[0]

                # non-parametric two-sample test between contra and ipsi stimulus (rank sum test)
                p_contra_ipsi = ranksums(stimulus_firing_rate_contra, stimulus_firing_rate_ipsi).pvalue[0]

                # compute mean firing rate for contra and ipsi stimulus
                mean_firing_rate_contra = np.mean(stimulus_firing_rate_contra)
                mean_firing_rate_ipsi = np.mean(stimulus_firing_rate_ipsi)
                
                # store p_contra, p_ipsi and p_contra_ipsi in a pandas data frame 
                # together with the cluster id, the eid, and x, y and z coordinates of the cluster
                cluster_df = pd.DataFrame({'cluster_id': clust, 'eid': eid, 'atlas_id': clusters_roi['atlas_id'][clusters_roi['cluster_id']==clust], 'x': clusters_roi['x'][clusters_roi['cluster_id']==clust], 'y': clusters_roi['y'][clusters_roi['cluster_id']==clust], 'z': clusters_roi['z'][clusters_roi['cluster_id']==clust], 'brain_area_label': area_label, 'mean_fr_contra': mean_firing_rate_contra, 'mean_fr_ipsi': mean_firing_rate_ipsi, 'p_contra': p_contra, 'p_ipsi': p_ipsi, 'p_contra_ipsi': p_contra_ipsi}, index=[0])

                # append cluster_df to the dataframe
                cluster_df_all = pd.concat([cluster_df_all, cluster_df], ignore_index=True)
                


        # avarage psth_contra across all clusters (column 0 is cluster id)
        mean_psth_contra_all_clusters = np.mean(psth_contra_all_clusters[:, 1:], axis=0) / 0.01  # divide by bin size to get firing rate in Hz
        # same for psth_ipsi
        mean_psth_ipsi_all_clusters = np.mean(psth_ipsi_all_clusters[:, 1:], axis=0) / 0.01  # divide by bin size to get firing rate in Hz

        # plot psth_contra and psth_ipsi
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 5))
        plt.plot(time_bins[:-1], mean_psth_contra_all_clusters, label='contra stimulus', color='blue')
        plt.plot(time_bins[:-1], mean_psth_ipsi_all_clusters, label='ipsi stimulus', color='red')
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.axvline(x=stimulus_window[0], color='black', linestyle='--')
        plt.axvline(x=stimulus_window[1], color='black', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
        # title with brain region and stimulus window
        plt.title(f'{brain_acronym} - {stimulus_window[0]} to {stimulus_window[1]} s')
        # add a legend
        plt.legend()
        # save plot to file as eps and png
        plt.savefig(f'png/{brain_acronym}_PSTH_{stimulus_window[0]}_{stimulus_window[1]}.png', format='png')
        plt.savefig(f'eps/{brain_acronym}_PSTH_{stimulus_window[0]}_{stimulus_window[1]}.eps', format='eps')

        # compute proportion of clusters that are significantly modulated by contra and ipsi stimuli
        contra_modulated = np.sum(np.bitwise_and(cluster_df_all['p_contra'] < 0.05, cluster_df_all['p_ipsi'] > 0.05)) / len(cluster_df_all)
        ipsi_modulated = np.sum(np.bitwise_and(cluster_df_all['p_contra'] > 0.05, cluster_df_all['p_ipsi'] < 0.05)) / len(cluster_df_all)
        print(f'Proportion of clusters modulated by contra stimulus: {contra_modulated:.2f}')
        print(f'Proportion of clusters modulated by ipsi stimulus: {ipsi_modulated:.2f}')

        # proportion of clusters that are significantly modulated by both contra and ipsi stimuli
        both_modulated = np.sum(np.bitwise_and(cluster_df_all['p_contra'] < 0.05, cluster_df_all['p_ipsi'] < 0.05)) / len(cluster_df_all)
        print(f'Proportion of clusters modulated by both contra and ipsi stimuli: {both_modulated:.2f}')

        # conduct one sample z-test for proportion to compare contra_modulated proportion to 0.05
        _, p_value_contra = proportions_ztest(contra_modulated*len(cluster_df_all), len(cluster_df_all), value=0.05,alternative='larger',prop_var=0.05)
        _, p_value_ipsi = proportions_ztest(ipsi_modulated*len(cluster_df_all), len(cluster_df_all), value=0.05,alternative='larger',prop_var=0.05)
        _, p_value_both = proportions_ztest(both_modulated*len(cluster_df_all), len(cluster_df_all), value=0.05,alternative='larger',prop_var=0.05)

        # statistically compare contra_modulated and ipsi_modulated proportions using binomial test
        n_contra = np.sum(np.bitwise_and(cluster_df_all['p_contra'] < 0.05, cluster_df_all['p_ipsi'] > 0.05))
        n_ipsi = np.sum(np.bitwise_and(cluster_df_all['p_contra'] > 0.05, cluster_df_all['p_ipsi'] < 0.05))
        p_contra_vs_ipsi = binomtest(n_contra, n_contra+n_ipsi, p=0.5, alternative='two-sided').pvalue

        # make a bar plot of the proportion of clusters modulated by contra and ipsi stimuli
        plt.figure(figsize=(5, 5))
        plt.bar(['Contra-responsive', 'Ipsi-responsive', 'Both-responsive'], [contra_modulated, ipsi_modulated, both_modulated], color=['blue', 'red', 'purple'])
        # add the numbers above each bar
        for i, v in enumerate([contra_modulated, ipsi_modulated, both_modulated]):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
        # add p_vavlue_contra, p_value_ipsi and p_value_both to the plot
        plt.text(0, 0.45, f'p-value contra: {p_value_contra:.4f}', ha='center', va='bottom', fontsize=8, color='blue')
        plt.text(1, 0.45, f'p-value ipsi: {p_value_ipsi:.4f}', ha='center', va='bottom', fontsize=8, color='red')
        plt.text(2, 0.45, f'p-value both: {p_value_both:.4f}', ha='center', va='bottom', fontsize=8, color='purple')
        # add p_value_contra_vs_ipsi to the plot (shift a bit down)
        plt.text(0.5, 0.35, f'p-value contra vs ipsi: {p_contra_vs_ipsi:.4f}', ha='center', va='bottom', fontsize=8, color='black')
        # draw a dashed horizontal line at y=0.05
        plt.axhline(y=0.05, color='black', linestyle='--')
        plt.ylabel('Proportion of clusters')
        # title with brain region and stimulus window
        plt.title(f'{brain_acronym} - {stimulus_window[0]} to {stimulus_window[1]} s')
        plt.ylim(0, 0.5)
        plt.xticks(rotation=45)
        plt.grid()
        # save plot to file as eps and png. Avoid cutting off the text by using bbox_inches='tight'
        plt.savefig(f'png/{brain_acronym}_Modulation_barplot_{stimulus_window[0]}_{stimulus_window[1]}.png', format='png', bbox_inches='tight')
        plt.savefig(f'eps/{brain_acronym}_Modulation_barplot_{stimulus_window[0]}_{stimulus_window[1]}.eps', format='eps', bbox_inches='tight')
        plt.show()

        # make a venn diagram of the proportions of clusters modulated by contra and ipsi stimuli
        subsets = (round(contra_modulated*100,2), round(ipsi_modulated*100,2), round(both_modulated*100,2))
        plt.figure(figsize=(5, 5))
        venn2(subsets=subsets, set_labels=('Contra-responsive', 'Ipsi-responsive'))
        # title with brain region and stimulus window
        plt.title(f'{brain_acronym} - {stimulus_window[0]} to {stimulus_window[1]} s')
        # save plot to file as eps
        plt.savefig(f'png/{brain_acronym}_Modulation_Venn_{stimulus_window[0]}_{stimulus_window[1]}.png', format='png')
        plt.savefig(f'eps/{brain_acronym}_Modulation_Venn_{stimulus_window[0]}_{stimulus_window[1]}.eps', format='eps')
        plt.show()

        # proportion of clusters with significant p_contra_ipsi
        # (i.e. significant difference between contra and ipsi stimuli)
        contra_ipsi_modulated = np.sum(cluster_df_all['p_contra_ipsi'] < 0.05) / len(cluster_df_all)
        print(f'Proportion of clusters modulated by contra versus ipsi stimuli: {contra_ipsi_modulated:.2f}')
        _, p_value_contra_ipsi_modulated = proportions_ztest(contra_ipsi_modulated*len(cluster_df_all), len(cluster_df_all), value=0.05,alternative='larger',prop_var=0.05)

        # add column to cluster_df_all with the difference between contra and ipsi mean firing rates
        cluster_df_all['mean_fr_diff'] = cluster_df_all['mean_fr_contra'] - cluster_df_all['mean_fr_ipsi']

        # what's the proportion of clusters with significant p_contra_ipsi and positive mean_fr_diff
        pos_mean_fr_diff = np.sum(np.bitwise_and(cluster_df_all['mean_fr_diff'] > 0, cluster_df_all['p_contra_ipsi'] < 0.05)) / len(cluster_df_all)
        print(f'Proportion of clusters with positive mean_fr_diff and significant p_contra_ipsi: {pos_mean_fr_diff:.2f}')

        # same for negative mean_fr_diff
        neg_mean_fr_diff = np.sum(np.bitwise_and(cluster_df_all['mean_fr_diff'] < 0, cluster_df_all['p_contra_ipsi'] < 0.05)) / len(cluster_df_all)
        print(f'Proportion of clusters with negative mean_fr_diff and significant p_contra_ipsi: {neg_mean_fr_diff:.2f}')

        # statistically compare pos_mean_fr_diff and neg_mean_fr_diff proportions using binomial test
        n_pos = np.sum(np.bitwise_and(cluster_df_all['mean_fr_diff'] > 0, cluster_df_all['p_contra_ipsi'] < 0.05))
        n_neg = np.sum(np.bitwise_and(cluster_df_all['mean_fr_diff'] < 0, cluster_df_all['p_contra_ipsi'] < 0.05))
        p_mean_fr_diff = binomtest(n_pos, n_pos+n_neg, p=0.5, alternative='two-sided').pvalue

        # make a bar plot of the proportion of clusters modulated by contra and ipsi stimuli
        plt.figure(figsize=(5, 5))
        plt.bar(['Side-selective', 'Contra-selective', 'Ipsi-selective'], [contra_ipsi_modulated, pos_mean_fr_diff, neg_mean_fr_diff], color=['green', 'blue', 'red'])
        # add the numbers above each bar
        for i, v in enumerate([contra_ipsi_modulated, pos_mean_fr_diff, neg_mean_fr_diff]):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
        # add p_value_contra_ipsi_modulated to the plot
        plt.text(0, 0.45, f'p-val. contra vs ipsi: {p_value_contra_ipsi_modulated:.4f}', ha='center', va='bottom', fontsize=8, color='green')
        # add p_mean_fr_diff to the plot, center it between the second and third bars
        plt.text(1.5, 0.35, f'p-val. mean_fr_diff: {p_mean_fr_diff:.4f}', ha='center', va='bottom', fontsize=8, color='black')
        # draw a dashed horizontal line at y=0.05
        plt.axhline(y=0.05, color='black', linestyle='--')
        plt.ylabel('Proportion of clusters')
        # title with brain region and stimulus window
        plt.title(f'{brain_acronym} - {stimulus_window[0]} to {stimulus_window[1]} s')
        plt.ylim(0, 0.5)
        plt.xticks(rotation=45)
        plt.grid()
        # save plot to file as eps
        plt.savefig(f'png/{brain_acronym}_Selectivity_barplot_{stimulus_window[0]}_{stimulus_window[1]}.png', format='png', bbox_inches='tight')
        plt.savefig(f'eps/{brain_acronym}_Selectivity_barplot_{stimulus_window[0]}_{stimulus_window[1]}.eps', format='eps', bbox_inches='tight')
        plt.show()


        # scatter plot of mean_fr_contra vs mean_fr_ipsi
        plt.figure(figsize=(5, 5))
        plt.scatter(cluster_df_all['mean_fr_contra'], cluster_df_all['mean_fr_ipsi'], alpha=0.5)
        plt.xlabel('Mean firing rate contra stimulus (Hz)')
        plt.ylabel('Mean firing rate ipsi stimulus (Hz)')
        plt.title('Mean firing rate contra vs ipsi stimulus')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axvline(x=0, color='black', linestyle='--')
        # add identity line (y=x)
        plt.plot([0, 100], [0, 100], color='black', linestyle='--')
        # title with brain region and stimulus window
        plt.title(f'{brain_acronym} - {stimulus_window[0]} to {stimulus_window[1]} s')
        plt.grid()
        # save plot to file as eps
        plt.savefig(f'png/{brain_acronym}_Response_scatterplot_{stimulus_window[0]}_{stimulus_window[1]}.png', format='png')
        plt.savefig(f'eps/{brain_acronym}_Response_scatterplot_{stimulus_window[0]}_{stimulus_window[1]}.eps', format='eps')
        plt.show()



# print unique atlas ids of clusters in the cluster_df_all dataframe
print('Unique atlas ids of clusters in the cluster_df_all dataframe:')
print(cluster_df_all['atlas_id'].unique())