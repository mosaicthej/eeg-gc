%% seed psuedochannel to target channel
% this version of the script paralellizes the actual PWGC computation
% across all seed -> target channels.  
% the 3 seed channels are first averaged together, to create a "psuedochannel" seed and then all pairwise GC are computed 
% to the target channels.

homeDir = fullfile(pwd(), '..'); % get fix-ed path here.
sourceDir = fullfile(homeDir,'GC'); % source dir is current directory
addpath(fullfile(sourceDir, 'bsmart/'));
% subjectID = [4001 4003 4004 4005 4006 4007 4008 4009 4010 4011 4013 4015 4016 4018 4020 4021 4022 4023];
% subjectID = [4001 4003 4004 4005 4006 4007 4008 4009 4010 4011 4013 4015 4016 4018 4020 4021 4022 4023];
% for some reason I was exluding 4004, 4020 but should exclude 4007, 4018 as that is what is exluded in SVM

% try ecluding 4, 14, 19 because >50% bad trials 
subjectID = [4001 4003 4005 4006 4007 4008 4009 4010 4011 4013 4015 4016 4018 4020 4021 4022 4023];
% subjectID = [4004 4014 4019]
% subjectID = [0004];

reref_scheme = 'mastoid_ref';
timing_scheme = 'BIOSEMI_analog_din';
preproc_scheme = 'eeglab_standard';


% with 500Hz fs % RUN THIS ASAP & visualize 
% define hyperparams for GC 
% Sri paper: sliding window 60ms (SW30, MO15)
% sliding_window = 16;
% model_order = 8;
% sliding_window = 30;
% model_order = 15;

% Jake paper: SW 40ms (SW20 MO10)
% sliding_window = 20;
% model_order = 10;

 sliding_window = 20;
 model_order = 10;
 fs=500;
 pop_thresh = 120;

% sliding_window = 60;
% model_order = 30;
% fs=2000;
% sliding_window = 120;
% model_order = 60;

% Jake paper: sliding window 40ms (SW20, MO10)
% sliding_window = 20;
% model_order = 10;
% fs=500;

% extra analysis for comparison: 20ms (SW10, MO5)
% sliding_window = 10;
% model_order = 5;
% fs=500;

% run PWGC over entire time window (entire time vector -300 to 500) 
% look into the subjects that have very high early PWGC (t=0) - perhaps
% noisy? why might that be? 
% 
if fs == 2000
    switch sliding_window
        case 60 % 30ms
            load('time_axis_SW30.mat');
            time_temp = [0:0.5:(302-sliding_window)];
            load('baseline_time_axis_SW30.mat');
            data_vector_length_ = length(time_temp);
            baseline_vector_length = 71;
        case 20
            load('time_axis_SW20.mat');
            load('baseline_time_axis_SW20.mat');
            data_vector_length = 132;
            baseline_vector_length = 81;
        case 15
            load('time_axis_SW15.mat');
            load('baseline_time_axis_SW15.mat');
            data_vector_length = 137;
            baseline_vector_length = 86;
    end

elseif fs == 500
    switch sliding_window
        case 15 % 30ms SW
            load('time_axis_SW15.mat');
            load('baseline_time_axis_SW15.mat');
            data_vector_length = 137;
            baseline_vector_length = 86;
        case 30 % 60ms SW
            load('time_axis_SW30.mat');
            load('baseline_time_axis_SW30.mat');
            data_vector_length_ = 122;
            baseline_vector_length = 71;
    end
end

% to compute new SW-specific time axis: 
% time_axis = times(time_window(1:end-SW+1))
%% load in the data - individual channels 

% compute GC moving window average over specific frequency range 
pwgc_results = cell(length(subjectID),1);

% nSeeds = 1;
% % nPsuedoChan = 1;
% % nTargets = 5;
% nTargets = 1;
% nSubjects = length(subjectID);
% pwgc_mov_seed2target_results = cell(nSeeds, nTargets, nSubjects);
% pairwise_combos = cell(nSeeds, nTargets, nSubjects);

tic
GC_derivDir = [homeDir, '/GC'];
cd(GC_derivDir)
load('ARO0004_ses-Sess1_task-AROEEG_0.1_30_sep_1_1_mastoid_ref_-0.2_0.5_500Hz_reSample_popthresh90.mat')
PWGC_analyses_type = 'full_epoch_window';
switch PWGC_analyses_type
    case 'full_epoch_window'
        fprintf('running PWGC on full epoch time windows.....\n')
        for subjIdx = 1:length(subjectID)
            subjName = ['ARO' int2str(subjectID(subjIdx))];
            fprintf('working on %s\n',subjName)
            derivDir = [homeDir, '/GC'];
            cd(derivDir)
            
            
            switch fs
                case 2000
                   if strcmp(reref_scheme, 'average_ref')
                       behavDataFile = sprintf(['ARO',int2str(subjectID(subjIdx)),'_task-perception_0.1_30_sep_1_1_average_ref_-0.2_0.6_popthresh%d.mat'],pop_thresh);
                   elseif strcmp(reref_scheme, 'mastoid_ref')
                       behavDataFile = sprintf(['ARO',int2str(subjectID(subjIdx)),'_task-perception_0.1_30_sep_1_1_mastoid_ref_-0.2_0.6_popthresh%d.mat'],pop_thresh);
                   end
                case 500
                     if strcmp(reref_scheme, 'average_ref')
                         behavDataFile = sprintf(['ARO' int2str(subjectID(subjIdx)) '_task-perception_0.1_30_sep_1_1_average_ref_-0.2_0.6_500Hz_reSample_popthresh%d.mat'],pop_thresh);
                     elseif strcmp(reref_scheme, 'mastoid_ref')
                         behavDataFile = sprintf(['ARO' int2str(subjectID(subjIdx)) '_task-perception_0.1_30_sep_1_1_mastoid_ref_-0.2_0.6_500Hz_reSample_popthresh%d.mat'],pop_thresh);
                     end
             end
                
            preproc_data = load('ARO0004_ses-Sess1_task-AROEEG_0.1_30_sep_1_1_mastoid_ref_-0.2_0.5_500Hz_reSample_popthresh90.mat');
            times = preproc_data.times;
            GC_derivDir = [homeDir, '/GC'];
            if ~isfolder(GC_derivDir)
                mkdir(GC_derivDir)
            end
            cd(GC_derivDir)
            
            %% select channels
            for xj = 1:64
                all_channels{xj} = preproc_data.chanlocs(xj).labels;
            end
%             % seed TEMPORAL 
% %             seed_pseudochan{1,1} = {'FT7','T7','TP7'}; %temporal
%             seed_pseudochan{1,1} = {'FCz','FC1','F1'}; %superior frontal
% %             % tartget seeds frontal and parietal 
% %             target_pseudochan{1,1} = {'F5','FC5','F3'}; %inferior frontal
% %             target_pseudochan{2,1} = {'FCz','FC1','F1'}; %superior frontal
% %             target_pseudochan{3,1} = {'Cz','C1','C3'}; %motor
%             target_pseudochan{1,1} = {'CPz','CP1','P1'}; %superior parietal
% %             target_pseudochan{5,1} = {'P5','P7', 'PO7'}; %inferior parietal
%             % Sri's orthographic sensors
% %             target_pseudochan{1,1} = {'P5','PO3','POz'};
% %             target_pseudochan{2,1} = {'C3','CP3','CP1','CPz','P1','P3','P5','Pz'};

            seed_pseudochan{1,1} = {'FT7','T7','TP7'}; % temporal
            seed_pseudochan{2,1} = {'F5','FC5','FC3'}; % inferior frontal
            seed_pseudochan{3,1} = {'FCz','FC1','F1'}; % superior frontal

            % tartgets
            target_pseudochan{1,1} = {'FT7','T7','TP7'}; % temporal
            target_pseudochan{2,1} = {'F5','FC5','FC3'}; % inferior frontal
            target_pseudochan{3,1} = {'FCz','FC1','F1'}; % superior frontal
            target_pseudochan{4,1} = {'CPz','CP1','P1'}; %superior parietal candidate #1
            target_pseudochan{5,1} = {'CP1','CP3','P1'}; %superior parietal candidate #2
            target_pseudochan{6,1} = {'CP5','P5','P7'}; % inferior parietal
            target_pseudochan{4,1} = {'Cz','C1','C3'}; % motor
            target_pseudochan{6,1} = {'CP1','CP3','P1'}; %superior parietal candidate #2
            target_pseudochan{7,1} = {'CP3','P1','P3'}; %superior parietal candidate #3
            target_pseudochan{8,1} = {'CP3','CP5','P3'}; %middle parietal
            target_pseudochan{9,1} = {'CP5','P5','P7'}; % inferior parietal
            target_pseudochan{10,1} = {'P5','PO3','POz'}; % sri orthographic sensor #1
            target_pseudochan{11,1} = {'C3','CP3','CP1','CPz','P1','P3','P5','Pz'}; % sri  parietal sensor
%{
            for xk = 1:length(seed_pseudochan)
                seed_idx = find(ismember(all_channels, seed_pseudochan{xk}));
                goodTrials = preproc_data.trialInfo;
                seed_pseudochan_data = preproc_data.data(seed_idx,:,goodTrials);
                num_windows = floor(size(seed_pseudochan_data, 2) / sliding_window);
                seed_window_avg = cell(1, num_windows);
                for w = 1:num_windows
                % Define start and end indices for each 20-point window
                    win_start = (w - 1) * sliding_window + 1;
                    win_end = w * sliding_window;
                    seed_window_avg{w} = mean(seed_pseudochan_data(:, win_start:win_end, :), 2);
                end
                seed_pseudochan_data_avg{xk,1} = mean(seed_pseudochan_data,1);
                clear seed_idx seed_pseudochan_data
            end
            
            for xm = 1:length(target_pseudochan)
                target_idx = find(ismember(all_channels,  target_pseudochan{xm}));
                goodTrials = preproc_data.trialInfo;
                target_pseudochan_data = preproc_data.data(target_idx,:,goodTrials);
                num_windows = floor(size(target_pseudochan_data, 2) / sliding_window);
                target_window_avg = cell(1, num_windows);
                for w = 1:num_windows
                    % Define start and end indices for each 20-point window
                    win_start = (w - 1) * sliding_window + 1;
                    win_end = w * sliding_window;

                    % Calculate the mean across this 20-point window
                    target_window_avg{w} = mean(target_pseudochan_data(:, win_start:win_end, :), 2);
                end

                target_pseudochan_data_avg{xm,1} = mean(target_pseudochan_data,1);
                clear target_idx target_pseudochan_data
            end
%}
            for xk = 1:length(seed_pseudochan)
                seed_idx = find(ismember(all_channels, seed_pseudochan{xk}));
                goodTrials = preproc_data.trialInfo;
                seed_pseudochan_data = preproc_data.data(seed_idx,:,goodTrials);
                seed_pseudochan_data_avg{xk,1} = mean(seed_pseudochan_data,1);
                clear seed_idx seed_pseudochan_data
            end
            
            for xm = 1:length(target_pseudochan)
                target_idx = find(ismember(all_channels,  target_pseudochan{xm}));
                goodTrials = preproc_data.trialInfo;
                target_pseudochan_data = preproc_data.data(target_idx,:,goodTrials);
                target_pseudochan_data_avg{xm,1} = mean(target_pseudochan_data,1);
                clear target_idx target_pseudochan_data
            end

%             %% Changing the lines above to add a sliding window of 20

%             %% separate the seed and target data into percDiff and prodDiff trials here 
%             % concatenate all trial orders into one array 
%             trialOrder_master = [];
%             try 
%                 for run = 1:5
%                     runIdx = sprintf('run%d',run);
%                     for trialNum = 1:100
%                     wordCode(trialNum) = preproc_data.EEG.etc. (runIdx).event(trialNum).wordCode;
%                     end
%                     trialOrder_master = [trialOrder_master;wordCode'];
%                     clear wordCode
%                 end
%             catch
%                 fprintf('trying with 4 runs..\n')
%                 trialOrder_master = [];
%                 for run = 1:4
%                     runIdx = sprintf('run%d',run);
%                     for trialNum = 1:100
%                         wordCode(trialNum) = preproc_data.EEG.etc. (runIdx).event(trialNum).wordCode;
%                     end
%                     trialOrder_master = [trialOrder_master;wordCode'];
%                     clear wordCode
%                 end
%             end
%             trialOrder_master_final = trialOrder_master;
%             trialOrder_master_final(preproc_data.EEG.trialInfo==0) = 0;
%             
%             %word key 
%             % /f/ 1 3 5 7 9
%             % /th/ 2 4 6 8 10
%             % /s/ 11 13 15 17 19
%             % /t/ 12 14 16 18 20
%             f_trials = find(ismember(trialOrder_master_final, [1,3,5,7,9]));
%             th_trials = find(ismember(trialOrder_master_final, [2,4,6,8,10]));
%             s_trials = find(ismember(trialOrder_master_final, [11,13,15,17,19]));
%             t_trials = find(ismember(trialOrder_master_final, [12,14,16,18,20]));
%             
%             prodDiff_f_th = [f_trials;th_trials];
%             percDiff_s_t = [s_trials;t_trials];
%             
%             for xxi = 1:length(seed_pseudochan_data_avg)
%                 seed_pseudochan_data_avg_percDiff{xxi,1} = seed_pseudochan_data_avg{xxi}(1,:,percDiff_s_t);
%                 seed_pseudochan_data_avg_prodDiff{xxi,1} = seed_pseudochan_data_avg{xxi}(1,:,prodDiff_f_th);
%             end
%             
%             for xxj = 1:length(target_pseudochan_data_avg)
%                 target_pseudochan_data_avg_percDiff{xxj,1} = target_pseudochan_data_avg{xxj}(1,:,percDiff_s_t);
%                 target_pseudochan_data_avg_prodDiff{xxj,1} = target_pseudochan_data_avg{xxj}(1,:,prodDiff_f_th);
%             end
            
            %% define time window for GC
            start_t = -200; %in samples
            end_t =  500; %in samples
            % find the index for the start and end time of interest
            diff1 = abs(times - start_t); [~,start_idx] = min(diff1);
            diff2 = abs(times - end_t); [~,end_idx] = min(diff2);
            time_window = [start_idx:end_idx];
            
            %% run stats
            
            % Temporary arrays for results
            nSeeds = length(seed_pseudochan_data_avg);
            nTargets = length(target_pseudochan_data_avg);
            temp_results = cell(nSeeds, nTargets);
            temp_combos = cell(nSeeds, nTargets);
            
            for s_idx = 1:nSeeds
                sdata = seed_pseudochan_data_avg{s_idx,:,:};
                seed_pseudochan_set = seed_pseudochan{s_idx};
                seed_combined = strjoin(seed_pseudochan_set, '_');

                for t_idx = 1:nTargets
                    if isequal(seed_pseudochan_set, target_pseudochan{t_idx})
                        % Write an empty cell array to the index if seed and target match
                        temp_results{s_idx, t_idx} = [];
                        temp_combos{s_idx, t_idx} = [];
                       continue;
                    end
                    target_pseudochan_set = target_pseudochan{t_idx};
                    target_combined = strjoin(target_pseudochan_set, '_');

                    dat = [sdata;target_pseudochan_data_avg{t_idx,:,:}];
                    datp1 = permute(dat, [2,1,3]); % now it is 350 x 2 x 655
                    datp2 = permute(dat, [2,3,1]); % or use 350 x 655 x 2
                    %             [Fxy, Fyx] = mov_bi_ga(dat,start_idx,end_idx,30,15,500,[1:30]);
                    tic
                    [Fxy, Fyx] = mov_bi_ga(datp1,start_idx,end_idx,sliding_window,model_order,500,[1:30]);
                    toc
                    pairwise_channel_combo = sprintf('seed_%s_target_%s', seed_combined, target_combined);
                    %             pwgc_mov_seed2target_results.(pairwise_channel_combo) = {Fxy, Fyx};
                    temp_results{s_idx, t_idx} = {Fxy, Fyx};
                    temp_combos{s_idx, t_idx} = pairwise_channel_combo;
                    %             temp{s_idx,t_idx} = pairwise_channel_combo;
                    %             pwgc_mov_seed2target_fn{s_idx,t_idx} = {pairwise_channel_combo, pairwise_channel_combo};
                    %             clear Fxy Fyx
                end
            end
            non_empty_indices = ~cellfun('isempty', temp_results);
            temp_results = temp_results(non_empty_indices);
            temp_combos = temp_combos(non_empty_indices);
            pwgc_mov_seed2target_results(:, :, subjIdx) = temp_results;
            pwgc_mov_seed2target_key(:,:, subjIdx) = temp_combos;
        end
        GC_results_dir = [GC_derivDir, 'seed_toto_target_channels/','pcROI_seed_to_pcROI_target/', ['SW' num2str(sliding_window) '_MO' num2str(model_order) '_fs' num2str(fs) '_3LHSeeds_LHTargets']];
%         GC_results_dir = [GC_derivDir, 'seed_to_target_channels/','pcROI_seed_to_pcROI_target/', ['SW' num2str(sliding_window) '_MO' num2str(model_order) '_fs' num2str(fs) '_LHTemporalSeed_SriOrthographic_Targets']];
        
        if ~isfolder(GC_results_dir)
            mkdir(GC_results_dir)
        end
        cd(GC_results_dir)
end


fname = sprintf('army_pwgc_1subj_test_Seeds_Targets_key.mat',fs,sliding_window,model_order);
% fname = sprintf('army_pwgc_1subj_test_Seeds_Targets_key.mat',fs,sliding_window,model_order,seed_combined);
fprintf('saving %s ....... \n\n', fname)
save(fname,'pwgc_mov_seed2target_results','-v7.3');

fname = sprintf('army_pwgc_1subj_test_Seeds_Targets_key.mat',sliding_window,model_order);
% fname = sprintf('army_pwgc_1subj_test_Seeds_Targets_key.mat',sliding_window,model_order,seed_combined);
fprintf('saving %s ....... \n\n', fname);
save(fname,'pwgc_mov_seed2target_key','-v7.3');
elapsed_time = toc; % Stop the timer and calculate the elapsed time
% Code after the point you want to measure
disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);

temp_combos = pwgc_mov_seed2target_key(:,:, 1);

%         
% create_perception_pwgc_figures_Fxy_v4(GC_results_dir,pwgc_mov_seed2target_key,squeeze(pwgc_mov_seed2target_results),seed_pseudochan,temp_combos,fs,sliding_window,model_order,pop_thresh)
% create_perception_pwgc_figures_Fyx_v4(GC_results_dir,pwgc_mov_seed2target_key,squeeze(pwgc_mov_seed2target_results),seed_pseudochan,temp_combos,fs,sliding_window,model_order,pop_thresh)
% 
% results_dir = ['/home/plamen/Research/SpeechProduction/EEG/derivatives/BSMART/perception/seed_to_target_channels/pcROI_seed_to_pcROI_target/SW20_MO10_fs500_3LHSeeds_LHTargets/python_visualization_dir'];
% perception_pwgc_data_to_python(results_dir,pwgc_mov_seed2target_key,squeeze(pwgc_mov_seed2target_results),seed_pseudochan,temp_combos,fs,sliding_window,model_order,pop_thresh)cd(GC_derivDir)  % bring us back to the point
cd(GC_derivDir)  % bring us back to the point
