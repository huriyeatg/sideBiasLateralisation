%%  this scripts creates the CSV session info for laser manipulations in DLS
clear all;
% For Huriye:
reversal_path = '\\Qnap-Al001\CToschi\DATA\reversal_learning_project\';
Cachefile_path = '\\Qnap-Al001\HAtilgan\Data\reversal_learning_project\cachefiles\';

C = table;
filename = 'SessionInfo_data_reversal.csv';
C = readtable([reversal_path filename],'FileType','text','Delimiter',',','ReadVariableNames',true,'TreatAsEmpty','');

t_sample_epoch = linspace(-0.5,1,100);
warp_sizes = [30,40,12,60]; %number of elements for each epoch: pre-stim, stim-choiceEnd, choiceEnd-outcome, post-outcome
zci = @(v) find(diff(sign(v))); % this is to figure out if the wheel crosses the zero point within the same trial

% Value2AFC = 2023-09-19_1_CHT005
% Value2AFC_Pavlov = 2023-12-19_1_CHT007
% Grating2AFC_Reversal = 2023-10-13_1_CHT005
% Grating2AFC_reversalCT_extraTrials_pavlov = 2023-12-18_1_CHT009

% clear all
% expRef = '2023-12-19_1_CHT009';
% behav2 = getBehavData(expRef, 'Grating2AFC_reversalCT_extraTrials_pavlov');
% blockFile = dat.loadBlock(expRef);

for a = 1:height(C)
    cacheFile = ([Cachefile_path, C.expRef{a} '.mat']);
    if ~exist(cacheFile,'file')
        fprintf('%d/%d %s\n',a, height(C),C.expRef{a});
        blockFile = dat.loadBlock(C.expRef{a});

        if matches(C.dataProfile{a}, 'Value2AFC')
            try behav2 = getBehavData(C.expRef{a}, 'Value2AFC');
            catch behav2 = getBehavData(C.expRef{a}, 'Value2AFC_noTimeline');
            end
            behav = behav2(:, {'trialNumber', 'repeatNumber', 'stimulusLeft', 'stimulusRight', 'choice', 'choiceCompleteTime', 'choiceStartTime', 'rewardVolume', 'punishSoundOnsetTime', 'stimulusOnsetTime', 'goCueTime', 'rewardTime'});

            % Convert categorical variables to cell array of character vector
            behav.stimulusLeft = cellstr(behav.stimulusLeft);
            behav.stimulusRight = cellstr(behav.stimulusRight);

            % Replace '<undefined>' with '1' and 'None' with '0' for 'stimulusLeft'

            behav.stimulusLeft(~strcmp(behav.stimulusLeft, 'None')) = {'1'};
            behav.stimulusLeft(strcmp(behav.stimulusLeft, 'None')) = {'0'};

            behav.proportionLeft = behav.stimulusLeft;

            % Replace '<undefined>' with '1' and 'None' with '0' for 'stimulusRight'
            behav.stimulusRight(~strcmp(behav.stimulusRight, 'None')) = {'1'};
            behav.stimulusRight(strcmp(behav.stimulusRight, 'None')) = {'0'};

            % Convert back to double
            behav.stimulusLeft = str2double(behav.stimulusLeft);
            behav.stimulusRight = str2double(behav.stimulusRight);

            StartTrial = blockFile.events.newTrialTimes(1:height(behav))';
            EndTrial = blockFile.events.endTrialTimes(1:height(behav))';
            behav.ITI = NaN(height(behav),1);
            behav.ITI(1:end-1, 1) = StartTrial(2:end,1) - EndTrial(1:end-1,1);
            behav.blockLength = NaN(height(behav),1);
            behav.accuracyThreshold = NaN(height(behav),1);
            behav.trialsToBuffer = NaN(height(behav),1);
            behav.extraTrials = NaN(height(behav),1);
            behav.TrialType = repmat({'Instrumental'}, height(behav), 1);
            behav.dataProfile = repmat({C.dataProfile{a}}, height(behav), 1);

        elseif matches(C.dataProfile{a}, 'Value2AFC_Pavlov')
            behav2 = getBehavData(C.expRef{a}, 'Value2AFC_Pavlov');
            behav = behav2(:, {'trialNumber', 'repeatNumber', 'stimulusLeft', 'stimulusRight', 'choice', 'choiceCompleteTime', 'choiceStartTime', 'rewardVolume', 'punishSoundOnsetTime', 'stimulusOnsetTime', 'goCueTime', 'rewardTime', 'TrialType'});

            % Convert categorical variables to cell array of character vector
            behav.stimulusLeft = cellstr(behav.stimulusLeft);
            behav.stimulusRight = cellstr(behav.stimulusRight);

            % Replace '<undefined>' with '1' and 'None' with '0' for 'stimulusLeft'

            behav.stimulusLeft(~strcmp(behav.stimulusLeft, 'None')) = {'1'};
            behav.stimulusLeft(strcmp(behav.stimulusLeft, 'None')) = {'0'};

            behav.proportionLeft = behav.stimulusLeft;

            % Replace '<undefined>' with '1' and 'None' with '0' for 'stimulusRight'
            behav.stimulusRight(~strcmp(behav.stimulusRight, 'None')) = {'1'};
            behav.stimulusRight(strcmp(behav.stimulusRight, 'None')) = {'0'};

            % Convert back to double
            behav.stimulusLeft = str2double(behav.stimulusLeft);
            behav.stimulusRight = str2double(behav.stimulusRight);

            StartTrial = blockFile.events.newTrialTimes(1:height(behav))';
            EndTrial = blockFile.events.endTrialTimes(1:height(behav))';
            behav.ITI = NaN(height(behav),1);
            behav.ITI(1:end-1, 1) = StartTrial(2:end,1) - EndTrial(1:end-1,1);
            behav.blockLength = NaN(height(behav),1);
            behav.accuracyThreshold = NaN(height(behav),1);
            behav.trialsToBuffer = NaN(height(behav),1);
            behav.extraTrials = NaN(height(behav),1);
            behav.dataProfile = repmat({C.dataProfile{a}}, height(behav), 1);

        elseif matches(C.dataProfile{a}, 'Grating2AFC_Reversal')
            behav2 = getBehavData(C.expRef{a}, C.dataProfile{a});
            blockFile = dat.loadBlock(C.expRef{a});
            behav2 = renamevars(behav2, {'contrastLeft', 'contrastRight'}, {'stimulusLeft', 'stimulusRight'});
            behav = behav2(:, {'trialNumber', 'repeatNumber', 'stimulusLeft', 'stimulusRight', 'choice', 'choiceCompleteTime', 'choiceStartTime', 'rewardVolume', 'punishSoundOnsetTime', 'stimulusOnsetTime', 'goCueTime', 'rewardTime', 'blockLength', 'accuracyThreshold', 'trialsToBuffer'});
            behav.proportionLeft = blockFile.events.proportionLeftValues(1:height(behav))';
            behav.extraTrials = NaN(height(behav),1);
            behav.TrialType = repmat({'Instrumental'}, height(behav), 1);
            StartTrial = blockFile.events.newTrialTimes(1:height(behav))';
            EndTrial = blockFile.events.endTrialTimes(1:height(behav))';
            behav.ITI = NaN(height(behav),1);
            behav.ITI(1:end-1, 1) = StartTrial(2:end,1) - EndTrial(1:end-1,1);
            behav.dataProfile = repmat({C.dataProfile{a}}, height(behav), 1);

        elseif contains(C.dataProfile{a}, 'Grating2AFC_reversalCT_extraTrials_pavlov')
            behav2 = getBehavData(C.expRef{a}, C.dataProfile{a});
            blockFile = dat.loadBlock(C.expRef{a});
            behav2 = renamevars(behav2, {'contrastLeft', 'contrastRight'}, {'stimulusLeft', 'stimulusRight'});
            behav = behav2(:, {'trialNumber', 'repeatNumber', 'stimulusLeft', 'stimulusRight', 'choice', 'choiceCompleteTime', 'choiceStartTime', 'rewardVolume', 'punishSoundOnsetTime', 'stimulusOnsetTime', 'goCueTime', 'rewardTime', 'TrialType', 'blockLength', 'accuracyThreshold', 'trialsToBuffer', 'extraTrials'});
            pavlovTrials = blockFile.paramsValues(1).pavlovTrials;
            behav.proportionLeft = NaN(height(behav),1);
            behav.proportionLeft(pavlovTrials+1:end) = blockFile.events.proportionLeftValues(1:height(behav)-pavlovTrials)';
            StartTrial = blockFile.events.newTrialTimes(1:height(behav))';
            EndTrial = blockFile.events.endTrialTimes(1:height(behav))';
            behav.ITI = NaN(height(behav),1);
            behav.ITI(1:end-1, 1) = StartTrial(2:end,1) - EndTrial(1:end-1,1);
            behav.dataProfile = repmat({C.dataProfile{a}}, height(behav), 1);

        else 
            disp('problem with DataProfile')
            continue
        end

        behav.expRef = repmat({C.expRef{a}}, height(behav), 1);

        behav.sessionNum = repmat(C.sessionNum(a), height(behav), 1);
        behav.reversal = repmat({C.reversal{a}}, height(behav), 1);
        numTrials = height(behav);
        behav.outcomeTime = nanmean([behav.rewardTime behav.punishSoundOnsetTime],2);

        %Get photometry data, align and zscore
        chanLabel = {C.photometryChannel1{a}, C.photometryChannel2{a},...
            C.photometryChannel3{a}, C.photometryChannel4{a}};

        %% fluorescence data
        fluor = table; %create blank table for fluorescence data

        %preallocate timestamps for warping
        warp_samples = nan(length(behav.choice), sum(warp_sizes));

        for tr = 1:length(behav.choice)
            epoch1 = linspace(behav.stimulusOnsetTime(tr)-0.5, behav.stimulusOnsetTime(tr), warp_sizes(1));
            epoch2 = linspace(behav.stimulusOnsetTime(tr), behav.choiceStartTime(tr), warp_sizes(2));
            epoch3 = linspace(behav.choiceStartTime(tr), behav.outcomeTime(tr), warp_sizes(3));
            epoch4 = linspace(behav.outcomeTime(tr), behav.outcomeTime(tr)+1, warp_sizes(4));
            warp_samples(tr,:) = [epoch1, epoch2, epoch3, epoch4];
        end

        if sum(cellfun(@(x) strcmp(x,'none'), chanLabel))<length(chanLabel) && (~any(contains(chanLabel, 'PFC'))) %if any channels have labels


            photometry = matfri_photometryAlign(C.expRef{a} , 'plot', false, 'numSecToDetrend', 25, 'alignWithRewards', strcmp(C.alignWithReward{a},'TRUE'));
            %photometry = photometryAlign(C.expRef{a} , 'plot', false, 'numSecToDetrend', 25, 'alignWithRewards', strcmp(C.alignWithReward{a},'TRUE'));

            %Get green channels only
            chans = {'channel1_0G','channel2_2G','channel3_4G','channel4_6G'}; %green filtered channels
            for y = 1:length(chanLabel)
                if ~contains(chanLabel{y}, 'none')
                    chan = y;
                    channel = strsplit(chanLabel{y},' - ');
                    dff = photometry.(chans{chan});

                    f1 = behav(:,{'expRef','trialNumber', 'sessionNum', 'reversal', 'rewardVolume', 'TrialType'});
                    f1.region = repmat({channel{1}},numTrials,1);
                    f1.neurotransmitter = repmat({channel{2}},numTrials,1);
                    f1 = alignToEvent('F', f1, dff, photometry.Timestamp, t_sample_epoch, warp_samples, behav.stimulusOnsetTime, behav.choiceStartTime, behav.outcomeTime);
                    fluor = [fluor; f1];
                end
            end
            fluor = sortrows(fluor,{'expRef','TrialType', 'trialNumber', 'region'});



            avrg_stim_to_choiceStart = [];
            avrg_stim_to_priorOutcome = [];
            avrg_choiceStart_priorOutcome = [];
            avrg_postOutcome = [];
            avrg_stim_to_postOutcome = [];


            for x = 1:height(fluor)
                pap = [];
                pap = fluor.F_timewarped(x, :);
                avrg_choiceStart_priorOutcome(x) = nanmean(pap(71:81),2);
                avrg_stim_to_priorOutcome(x) = nanmean(pap(31:81),2);
                avrg_stim_to_choiceStart(x) = nanmean(pap(31:70),2);
                avrg_postOutcome(x) = nanmean(pap(83:end),2);
                avrg_stim_to_postOutcome(x) = nanmean(pap(31:end),2);
            end


            fluor.avrg_stim_to_choiceStart = avrg_stim_to_choiceStart';
            fluor.avrg_choiceStart_priorOutcome = avrg_choiceStart_priorOutcome';
            fluor.avrg_stim_to_priorOutcome = avrg_stim_to_priorOutcome';
            fluor.avrg_postOutcome = avrg_postOutcome';
            fluor.avrg_stim_to_postOutcome = avrg_stim_to_postOutcome';

        end

        %% wheel data


        %take data on wheel
        wheelpos = blockFile.inputs.wheelMMValues;
        wheelt = blockFile.inputs.wheelMMTimes;


        %smooth and convert to velocity
        dt = median(diff(wheelt));
        smooth_t = 10/1000;  %10ms smoothing
        smoothWin = myGaussWin(smooth_t, 1/dt);
        wheelvel = conv( diff(wheelpos)/dt, smoothWin, 'same');


        wheel = table;
        wheel = behav(:,{'expRef','trialNumber', 'sessionNum', 'reversal'});
        wheel = alignToEvent('wheelVel', wheel, wheelvel, wheelt(2:end), t_sample_epoch, warp_samples, behav.stimulusOnsetTime, behav.choiceStartTime, behav.outcomeTime);
        wheel = alignToEvent('wheelPosition', wheel, wheelpos, wheelt, t_sample_epoch, warp_samples, behav.stimulusOnsetTime, behav.choiceStartTime, behav.outcomeTime);

        cross_zero = cell(1, height(wheel));
        Cross_zero = 0;

        for x = 1:height(wheel)

            cross_zero{x}  = zci(wheel.wheelPosition_timewarped(x, 31:70));
            if isempty(cross_zero{x})
                Cross_zero(x) = 0; %wheel doesn't cross the center point, no giggle
            else
                Cross_zero(x) = 1; %wheel crosses the center point,  giggle
            end
        end

        %%

        %flip the velocity sign for Right choices
        flipIdx = behav.choice=='Right';

        wheel.wheelVel_stim_flipped = wheel.wheelVel_stim;
        wheel.wheelVel_choice_flipped = wheel.wheelVel_choice;
        wheel.wheelVel_outcome_flipped = wheel.wheelVel_outcome;
        wheel.wheelVel_timewarped_flipped = wheel.wheelVel_timewarped;

        wheel.wheelPosition_stim_flipped = wheel.wheelPosition_stim;
        wheel.wheelPosition_choice_flipped = wheel.wheelPosition_choice;
        wheel.wheelPosition_outcome_flipped = wheel.wheelPosition_outcome;
        wheel.wheelPosition_timewarped_flipped= wheel.wheelPosition_timewarped;

        %now flip
        wheel.wheelVel_stim_flipped(flipIdx,:) = -wheel.wheelVel_stim(flipIdx,:);
        wheel.wheelVel_choice_flipped(flipIdx,:) = -wheel.wheelVel_choice(flipIdx,:);
        wheel.wheelVel_outcome_flipped(flipIdx,:) = -wheel.wheelVel_outcome(flipIdx,:);
        wheel.wheelVel_timewarped_flipped(flipIdx,:) = -wheel.wheelVel_timewarped(flipIdx,:);

        wheel.wheelPosition_stim_flipped(flipIdx,:) = -wheel.wheelPosition_stim(flipIdx,:);
        wheel.wheelPosition_choice_flipped(flipIdx,:) = -wheel.wheelPosition_choice(flipIdx,:);
        wheel.wheelPosition_outcome_flipped(flipIdx,:) = -wheel.wheelPosition_outcome(flipIdx,:);
        wheel.wheelPosition_timewarped_flipped(flipIdx,:) = -wheel.wheelPosition_timewarped(flipIdx,:);

        wheel.Cross_zero = Cross_zero';



        save(cacheFile, 'behav', 'wheel','fluor');

    end
end



%% Prepare data: Combine cached sessions into mega tables
clear all; close all;

C = table;
filename = 'SessionInfo_data_reversal.csv';
reversal_path = 'D:\LakLab project\data\data-dls\';
C = readtable([reversal_path filename],'FileType','text','Delimiter',',','ReadVariableNames',true,'TreatAsEmpty','');

export_path = 'D:\LakLab project\analysis\new file\';
Cachefile_path = 'D:\LakLab project\analysis\cachefile\';

D = cell( height(C), 1); %behavioural data
F = cell( height(C), 1); %fluorescence data
W = cell( height(C), 1); %wheel data
% E = cell( height(p.sessionInfo), 1); %eye data
% FA = cell( height(p.sessionInfo), 1); %face data

for sess = 1:height(C)
    fprintf('%d/%d\n',sess, height(C));
    cacheFile = fullfile(Cachefile_path, [C.expRef{sess} '.mat']);
    dat=load(cacheFile);

    if iscell(dat.behav.proportionLeft)
        dat.behav.proportionLeft = str2double(dat.behav.proportionLeft);
    else
    end
    %D{sess} = dat.behav;
    F{sess} = dat.fluor;
    %W{sess} = dat.wheel;
    %     E{sess} = dat.eye;
    %     FA{sess} = dat.face;

end

% strList = ["CHT007", "CHT008", "CHT009", "CHT010", "CHT011", "CHT012"];
% for sess = 1:height(C)
%     fprintf('%d/%d\n',sess, height(C));
%     cacheFile = fullfile(Cachefile_path, [C.expRef{sess} '.mat']);
%     if exist(cacheFile,'file')
%     dat=load(cacheFile);
%
%     if   any(contains(dat.behav.expRef{1}, strList))
%
%         if iscell(dat.behav.proportionLeft)
%             dat.behav.proportionLeft = str2double(dat.behav.proportionLeft);
%         else
%         end
%         D{sess} = dat.behav;
%         F{sess} = dat.fluor;
%         %W{sess} = dat.wheel;
%         %     E{sess} = dat.eye;
%         %     FA{sess} = dat.face;
%
%         %a=a+1;
%     else
%     end
%     else
%     end
% end


%D = D(~cellfun(@isempty, D));
% D = cat(1,D{:});
% D = sortrows(D,{'expRef','trialNumber'});

%W = cat(1,W{:});
%W = sortrows(W,{'expRef','trialNumber'});

%F = F(~cellfun(@isempty, F));
F = cat(1,F{:});
F = sortrows(F,{'expRef','trialNumber','region'});

%E = cat(1,E{:});
%E = sortrows(E,{'expRef','trialNumber'});
%Eye preprocessing not currently working

%FA = cat(1,FA{:});
%FA = sortrows(FA,{'expRef','trialNumber'});
%Face preproccesing not currently working

writetable(D, [export_path 'ReversalProject_behaviour3.csv']);
%writetable(W, [export_path 'ReversalProject_wheel2CHTGPT.csv']);
writetable(F, [export_path 'ReversalProject_fluo3.csv']);



function tableOut = alignToEvent(label, tableIn, x, t, t_sample_epoch, warp_samples, stimOnTime, choiceTime, outcomeTime)
% This function simply computes the event-aligned values for the signal x
% (at timestamps t) for stimulus, choice, and outcome times. Also does
% baseline correction.

tableOut = tableIn;

tableOut.([label '_stim_noSub']) = single(interp1(t,x,stimOnTime + t_sample_epoch));
tableOut.([label '_choice_noSub']) = single(interp1(t,x,choiceTime + t_sample_epoch));
tableOut.([label '_outcome_noSub'])  = single(interp1(t,x,outcomeTime + t_sample_epoch));
tableOut.([label '_timewarped_noSub']) = single(interp1(t,x,warp_samples));

tableOut.([label '_stim']) = single(interp1(t,x,stimOnTime + t_sample_epoch));
tableOut.([label '_choice']) = single(interp1(t,x,choiceTime + t_sample_epoch));
tableOut.([label '_outcome'])  = single(interp1(t,x,outcomeTime + t_sample_epoch));
tableOut.([label '_timewarped']) = single(interp1(t,x,warp_samples));

%baseline correct
pre_stim_baseline = mean(tableOut.([label '_stim'])(:,t_sample_epoch<0,:),2,'omitnan');
tableOut.([label '_stim']) = tableOut.([label '_stim']) - pre_stim_baseline;
tableOut.([label '_choice']) = tableOut.([label '_choice']) - pre_stim_baseline;
tableOut.([label '_outcome']) = tableOut.([label '_outcome']) - pre_stim_baseline;
tableOut.([label '_timewarped']) = tableOut.([label '_timewarped']) - pre_stim_baseline;

%calculate DA release according to pre-outcome baseline
pre_outcome_baseline = mean(tableOut.([label '_outcome'])(:,t_sample_epoch<0,:),2,'omitnan');
tableOut.([label '_outcome_PreOutcomeBaseline']) = tableOut.([label '_outcome']) - pre_outcome_baseline;


end




