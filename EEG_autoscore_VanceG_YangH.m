clear
fclose all;
tic
%% Ensemble Autoscoring of EDF %%
% read .edf files exported by PAL 8200 or PAL 8400
% auto-scores the recording based on training socres (~3 hrs of recording)
% Combination of LDA, SVM, NB, NN, Random-Subspace kNN, Random-Subspace Tree,
% and Bagged Tree
% writes autoscores into new file

%Vance Gao (https://github.com/gaodifan/Automatic-sleep-scoring)
%Last edited 2018-03-13
%Last adopted by Yang Hu 2018-8-23 to Sirenia Sleep - 1.7.8

% ------------------------------------------------------------------------

% MANUALLY SET THESE VALUES:

fileName = 'C:\Users\yhu1\Documents\MATLAB\Yang (E43_47)_8200.edf';

% TSV TRAINING SCORE:
TSV_fname = 'C:\Users\yhu1\Documents\MATLAB\Yang (E43_47)2.tsv';

% if sampling rate is high, may keep one sample per X samples to speed up 
% FFT and reduce memory:
skip = 10; 

% Fraction of epochs to reject:
rejFrac = 0.05;

% Which EDF channel contains the signals we want to be analyzed:
channel_EEG = 4; %eeg1=4, eeg2=5
channel_EMG = 6;

%% Read the EDF file

commandwindow
% .EDF file structure, using hex reader (PAL 8200 format, 1000 Hz):
% byte 0-2047: Header
% byte 2048-2051: unix time (4 bytes, 2051 2050 2049 2048)
% byte 2052-2053: gain (almost always 1)
% byte 2054-2055: 1st epoch score
% byte 2056-22055: 1st epoch EEG1
% byte 22056-42055: 1st epoch EEG2
% byte 42056-62055: 1st epoch EMG
% byte 62056-62855: zeroes (annotations)
% byte 62856-62859: time
% byte 62860-62861: gain
% byte 62862-62863: 2nd epoch score
% byte 62864-etc.: 2nd epoch EEG1...

% data is little-endian, 16-bit twos-complement
% Read into Matlab, bytes -> 16-bit integers ('short'): 
% Matlab short index *2 -2 = byte index (first of two bytes)

fileID = fopen(fileName);

% read recording info in header
headerBuf = fread(fileID,100000);
frewind(fileID);

headerLength = eval( char( headerBuf(185:192))) / 2; %in shorts
nEpochs = eval (char( headerBuf(237:244)));
secPerEp =  eval (char( headerBuf(245:252)));
nSignals = eval (char( headerBuf(253:256)));

ind = 257 + 216 * nSignals;
for s = 1:nSignals
    signalLengths(s) = eval( char( headerBuf( (ind:(ind+7)) + (s-1)*8)));
end
epShortsLength = sum(signalLengths);

% user check
disp(' ')
disp('-------------------------------------------------------------------')
disp(' ')
disp(datetime('now'))
disp(['Autoscoring file ' '''' fileName '''']);
disp(' ')
disp(['Percent of epochs to be rejected: ' num2str(rejFrac * 100) '%'])
disp(' ')
disp('Full list of signal channels:')

for s=1:nSignals
    disp([' ' num2str(s) '. ' char( headerBuf((257:272)+16*(s-1)))'])
end

disp(' ')
disp('Taking the following signal channel numbers as input:')
disp([' EEG: ' num2str(channel_EEG) '  EMG: ' num2str(channel_EMG)])
if any([channel_EEG > nSignals, channel_EMG > nSignals])
    error('*ERROR*: Assigned channel numbers are higher than number of channels! \n%s',...
         'Please assign correct signal channels in lines 27-29.')   
else
    disp('If channel numbers do not match, change settings to correct signal channels in lines 27-29.') 
end
disp(' ')

trueSampPerEp = signalLengths(channel_EEG);
trueSampRate = trueSampPerEp / secPerEp;
Hz = trueSampRate/skip;
sampPerEp = numel(1:skip:trueSampPerEp);

disp(['True sampling rate: ' num2str(trueSampRate)])
disp(['Skip: ' num2str(skip)])
disp(['Effective sampling rate: ' num2str(Hz)])
disp(' ')

disp('Does setup look correct? Ctrl+C if no, any key if yes:');
pause

% pre-allocate recording structure
recording.eeg2 = zeros(nEpochs, sampPerEp);
recording.emg = zeros(nEpochs, sampPerEp);

% ------------------------------------------------------------------------
%% Read the training score in TSV file
% Modified 2018-07-20 by Yang Hu
% GET: trainingscores (recording.scores), headers, data (without_headers)
% Check TSV trainging original file structure, using MATLAB double-click open:
% TSV is TAB seperated data file, also use textscan with 'Delimiter \t' to check the location of the TABs.
% lines 1-11: headers 
%(line No.10 is a blank newline structured by a blank space,line10 is omitted after textscan)
% lines 12 and on: data columns of date, time , time started (num), accumulated Epoch(num), score(num)
disp(' ')
disp('Reading the training score TSV file...')

% seperate headers from data
fileRead_formatSpec = '%s';
fid = fopen (TSV_fname,'rt');
read_txt = textscan(fid,fileRead_formatSpec, 'WhiteSpace', '\r', 'TextType', 'string',  'ReturnOnError', false);
read_txt2 = read_txt{1,1};
headers = read_txt2 (1:10,1);
data = read_txt2 (11:length (read_txt2),1);

% extract training scores
% preallocation
training_scores = zeros(nEpochs,1);
ii = 0;
for ii = 1 : nEpochs
data_in_num = str2num(char(data(ii,1)));
training_scores (ii,1) = data_in_num (end);
end
recording.scores = training_scores;%1 Wake; 2 NREM; 3 REM; 129 Wake-Ex; 130 NREM-Ex; 131 REM-Ex; 0 Artifact; 255 Unscored 
% ------------------------------------------------------------------------

% read file
disp(' ')
disp('Reading the training score file...')
disp(['- Shorts in header: ' num2str(headerLength)]);
disp(['- Shorts per epoch: ' num2str(epShortsLength)]);
disp(' ');

fread( fileID, headerLength, '*int16');

for k=1:nEpochs
    if mod(k,1000)==0
        disp(['Reading epoch ' num2str(k) '/' num2str(nEpochs)]);
    end
    
    epochBuf = fread( fileID, epShortsLength, '*int16');
    
    ind = sum( signalLengths(1:channel_EEG-1)) + 1; 
    recording.eeg2(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);      
    
    ind = sum( signalLengths(1:channel_EMG-1)) + 1; 
    recording.emg(k,:) = epochBuf(ind: skip: ind+signalLengths(channel_EEG)-1);
    
end
fclose(fileID);

clear k j s fileID epochBuf fileInfo skip headerBuf ind

%% Power Spectral Density for EEG2 and EMG, hour by hour
disp(' ')
disp('Performing short-time Fourier transform:') 

numHrs = ceil(nEpochs/360);

nfft = 2^nextpow2(sampPerEp);

eeg2P = zeros(ceil((nfft+1)/2), nEpochs);
emgP = zeros(ceil((nfft+1)/2), nEpochs);

for hr = 1:numHrs
    disp(['FFT: hour ' num2str(hr) '/' num2str(ceil(numHrs))])
    
    if hr ~= numHrs
        eeg2Vect = reshape(recording.eeg2(1+(hr-1)*360: hr*360, :)', sampPerEp*360, 1);
        emgVect = reshape(recording.emg(1+(hr-1)*360: hr*360, :)', sampPerEp*360, 1);        
        
        [~,eeg2F,~,eeg2Pbuf] = spectrogram(eeg2Vect, sampPerEp, 0, nfft, Hz);
        [~,emgF,~,emgPbuf] = spectrogram(emgVect, sampPerEp, 0, nfft, Hz);
        
        eeg2P(:, 1+(hr-1)*360: hr*360) = eeg2Pbuf;
        emgP(:, 1+(hr-1)*360: hr*360) = emgPbuf;
        
    else
        eeg2Vect = reshape(recording.eeg2(1+(hr-1)*360: end, :)', sampPerEp*(nEpochs-(hr-1)*360), 1);
        emgVect = reshape(recording.emg(1+(hr-1)*360: end, :)', sampPerEp*(nEpochs-(hr-1)*360), 1);
        
        [~,eeg2F,~,eeg2Pbuf] = spectrogram(eeg2Vect, sampPerEp, 0, nfft, Hz);
        [~,emgF,~,emgPbuf] = spectrogram(emgVect, sampPerEp, 0, nfft, Hz);
        
        eeg2P(:, 1+(hr-1)*360: end) = eeg2Pbuf;
        emgP(:, 1+(hr-1)*360: end) = emgPbuf;
    end
end

trainingScores = recording.scores;

% calculate root-mean-squares for later artifact detection
eegRMS = zeros(nEpochs,1);
for n = 1:nEpochs
    eegRMS(n) = rms(recording.eeg2(n,:));
end

clear eeg2Pbuf emgPbuf eeg2Vect emgVect hr nfft sampPerRec
recording = rmfield(recording,{'emg', 'eeg2'});


%% Pre-Classification Processing, Feature Extraction
% split data into logarithmic frequency bands

bands = logspace(log10(0.5), log10(Hz/2), 21);
input = zeros(nEpochs, 20);

for  i=1:20 
    input(:,i) = sum(eeg2P(and(eeg2F>=bands(i), eeg2F<bands(i+1)),:), 1)';
end
input(:,21) = sum(emgP(and(emgF>=4, emgF<40),:));

% normalize using a log transformation and smooth over time
input = conv2(log(input), fspecial('gaussian',[5 1],0.75),'same');

%% ------------------------------------------------------------------------
% clean the  'empty' values from the input data(each band), turn them to -1
% These are excluded from trainSet and set to unscored in the end
% modified by Yang Hu

% pre-allocate recording structure
Valid_trainSet = zeros(size (input,1), size (input,2));
nEmpty_EEG_EMG = 0;
for y = 1: size (input,1)
for h = 1: size (input,2)
    Valid_trainSet(y,h) = isfinite (input(y,h)) == 1;
    if isfinite (input(y,h)) == 0
        input (y,h)= -1; % modified
        nEmpty_EEG_EMG = nEmpty_EEG_EMG + 1;
    end
end
end
Valid_trainSet2 = sum (Valid_trainSet')';
% ------------------------------------------------------------------------
%% clean up artifacts in training scores using outlier fence criteria
iqr_wake = quantile(eegRMS, 0.75) - quantile(eegRMS, 0.25); 
highFence_wake = quantile(eegRMS, 0.75) + 3*iqr_wake;
lowFence_wake = quantile(eegRMS, 0.25) - 3*iqr_wake;

iqr_NREM = quantile(eegRMS, 0.75) - quantile(eegRMS, 0.25); 
highFence_NREM = quantile(eegRMS, 0.75) + 3*iqr_NREM;
lowFence_NREM = quantile(eegRMS, 0.25) - 3*iqr_NREM;

iqr_REM = quantile(eegRMS, 0.75) - quantile(eegRMS, 0.25); 
highFence_REM = quantile(eegRMS, 0.75) + 3*iqr_REM;
lowFence_REM = quantile(eegRMS, 0.25) - 3*iqr_REM;

for k=1:13:nEpochs
    if trainingScores(k)==1
        if eegRMS(k) > highFence_wake || eegRMS(k) < lowFence_wake
            trainingScores(k) = 129; %Wake-X
        end
        
    elseif trainingScores(k)==2
        if eegRMS(k) > highFence_NREM || eegRMS(k) < lowFence_NREM
            trainingScores(k) = 255;
        end   
        
    elseif trainingScores(k)==3
        if eegRMS(k) > highFence_REM || eegRMS(k) < lowFence_REM
            trainingScores(k) = 255;
        end
    end
end

clear i


%% Classification by Individual Methods
% Keep reproducibility of every auto-scoring by setting the pseudorandom at same starting point
% see marked as modified (by Yang Hu)
disp(' ')
disp('Auto-scoring...')

trainSet = trainingScores==1 | trainingScores==2 | trainingScores==3 & Valid_trainSet2 == size(Valid_trainSet,2);% exclude invalid signals from training

% Decision Tree
disp('  Decision Tree')
rng(1); % modified
model = fitctree(input(trainSet,:), num2str(trainingScores(trainSet)));
[autoScoresDt, scoreDT] = predict(model, input);
autoScoresDt = str2num(autoScoresDt);

% KNN
disp('  k-Nearest Neighbors')
rng(1); % modified
model = fitcknn(input(trainSet,:),trainingScores(trainSet));
[autoScoresKnn, scoreKnn] = predict(model, input);   

% Linear Discriminant Analysis
disp('  Linear Discriminant Analysis')
rng(1); % modified
LDAmodel = fitcdiscr(input(trainSet,:), trainingScores(trainSet));
[autoScoresLda, scoreLda] = predict(LDAmodel,input);

% Naive Bayes
disp('  Naive Bayes')
rng(1); % modified
model = fitcnb(input(trainSet,:),trainingScores(trainSet));
[autoScoresNb, scoreNb] = predict(model,input);


% Support Vector Machine
disp('  Support Vector Machine')
  % one against all
rng(1); % modified
trainingScoresSvm = trainingScores;
trainingScoresSvm(trainingScoresSvm==2 | trainingScoresSvm==3) = 4;
SVM = fitcsvm(input(trainSet,:),trainingScoresSvm(trainSet),'BoxConstraint',1);
SVM = fitPosterior(SVM);
[autoScoresW, scoreSvmW] = predict(SVM,input);

trainingScoresSvm = trainingScores;
trainingScoresSvm(trainingScoresSvm==1 | trainingScoresSvm==3) = 4;
SVM = fitcsvm(input(trainSet,:),trainingScoresSvm(trainSet),'BoxConstraint',1);
SVM = fitPosterior(SVM);
[autoScoresN, scoreSvmN] = predict(SVM,input);

trainingScoresSvm = trainingScores;
trainingScoresSvm(trainingScoresSvm==1 | trainingScoresSvm==2) = 4;
SVM = fitcsvm(input(trainSet,:),trainingScoresSvm(trainSet),'BoxConstraint',1);
SVM = fitPosterior(SVM);
[autoScoresR, scoreSvmR] = predict(SVM,input);

  % confidence scores
scoreSvm = [scoreSvmW(:,1) scoreSvmN(:,1) scoreSvmR(:,1)];
for k = 1:nEpochs
    scoreSvm(k,:) = scoreSvm(k,:) / sum(scoreSvm(k,:));
end

[~,autoScoresSvm] = max(scoreSvm,[],2);

clear autoScoreSVMwake autoScoreNR autoScoreWN autoScoreWR trainingScoresWake
clear LDAmodel bayesModel SVM


% Neural Network
scoreNn(1:nEpochs,3) = 0;
%{
target(recording.scores==1,1) = true;
target(recording.scores==2,2) = true;
target(recording.scores==3,3) = true;

% create a pattern recognition network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% setup division of data for training and validation
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 50/100;
net.divideParam.testRatio = 0/100;

% train the network
[net,~] = train(net,input(trainSet,:)',target(trainSet,:)');

% classify
y = net(input');
[~, autoScoresNn] = max(y);
autoScoresNn = autoScoresNn';

scoreNn = y';
%}


%% Classification by Ensemble Methods

% Bagged Decision Tree
disp('  Bagged Decision Tree (ensemble of 100)')
rng(1); % For reproducibility
model = fitensemble(input(trainSet,:), num2str(trainingScores(trainSet)),'Bag',100,'tree','type','classification');
[autoScoresDtBag, scoreDtBag] = predict(model,input);
autoScoresDtBag = str2num(autoScoresDtBag);

% Random Subspace Decision Tree and k-NN
disp('  Random Subspace Tree and k-NN (ensemble of 100):')

outputDtRS = zeros(nEpochs,100);
outputKnnRS = zeros(nEpochs,100);

for learner = 1:100
    if mod(learner,5) == 0
        disp(learner)
    end
    seed = RandStream('mlfg6331_64'); % modified

    subspace = randsample(seed, 20, 10, false);
    
    model = fitcknn(input(trainSet,[subspace' 21]), trainingScores(trainSet));
    outputKnnRS(:,learner) = predict(model, input(:,[subspace' 21]));   
    
    model = fitctree(input(trainSet,[subspace' 21]), num2str(trainingScores(trainSet)));
    autoScoresBuf = predict(model, input(:,[subspace' 21]));
    outputDtRS(:,learner) = str2num(autoScoresBuf);    
end

autoScoresKnnRS = mode(outputKnnRS,2);
autoScoresDtRS = mode(outputDtRS,2);

scoreKnnRS = [sum(outputKnnRS==1,2) sum(outputKnnRS==2,2) sum(outputKnnRS==3,2)] /100;
scoreDtRS = [sum(outputDtRS==1,2) sum(outputDtRS==2,2) sum(outputDtRS==3,2)] /100;

disp('  Ensembles done.')

clear outputTree outputKNN subspace model knn treeModel learner training


%% Classifier Fusion

% consensus vote on confidence scores
scoreCons = (scoreDtBag +scoreDtRS +scoreKnnRS + scoreLda + scoreNb +scoreSvm +scoreNn) /7;
[~,autoScoresCons] = max(scoreCons,[],2);

% rejection criteria
autoScores = autoScoresCons;

maxConfScores = max(scoreCons, [], 2);

confOrder = tiedrank(maxConfScores);
rejectSet = confOrder < rejFrac*nEpochs;
autoScores(rejectSet) = 255;


%% Cleanup

% 1) Artifact removal by power thresholds
nExcludeNR = 0;
nExcludeW = 0;

if rejFrac ~= 0
    for k=1:nEpochs  
        if autoScores(k)==1 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_wake || eegRMS(k) < lowFence_wake
                autoScores(k)= 129; %Wake-X
                nExcludeW = nExcludeW+1;
            end

        elseif autoScores(k)==2 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_NREM || eegRMS(k) < lowFence_NREM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR+1;
            end   

        elseif autoScores(k)==3 && ~ismember(trainingScores(k),[1 2 3 129 130 131])
            if eegRMS(k) > highFence_REM || eegRMS(k) < lowFence_REM
                autoScores(k)= 255;
                nExcludeNR = nExcludeNR + 1;
            end
        end
    end
end

%------------------------------------------------------------------------
% calculate how many overrides have been unscored (modified by Yang Hu)
nUnscored_in_Override= 0;

for u=1:nEpochs  
     
     if ismember(trainingScores(u), [0 1 2 3 129 130 131]) && autoScores(u)== 255
        nUnscored_in_Override = nUnscored_in_Override + 1;
        end  
end
%------------------------------------------------------------------------

% 2) Manual training scores override auto-scores

nOverride = 0;

for k=1:nEpochs  
     
     if ismember(trainingScores(k), [0 1 2 3 129 130 131]) && trainingScores(k)~=autoScores(k)
        autoScores(k) = trainingScores(k);
        nOverride = nOverride + 1;
        end  
end

% 3) Improbable sequences set to unscored
%
if rejFrac ~= 0
    autoScoresBuf = autoScores;

    for k = 2:nEpochs 
        if autoScores(k-1)==1 && autoScores(k)==3 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end

        if autoScores(k-1)==3  && autoScores(k)==2 ...
                && ~ismember(trainingScores(k), [1 2 3 129 130 131]) ...
                && ~ismember(trainingScores(k-1), [1 2 3 129 130 131])
            autoScores(k) = 255;
            autoScores(k-1) = 255;
        end
    end
end
% ------------------------------------------------------------------------
% introduce autoscores and set false EMG signals to unscored (modified by Yang Hu)

misEpoch = 0;
for fEMG = 1:length (Valid_trainSet2)
    if Valid_trainSet2 (fEMG) ~= size(Valid_trainSet,2)
        autoScores(fEMG) = 255;
        misEpoch = misEpoch + 1;
    end
end
% ------------------------------------------------------------------------
nImprobable = sum(autoScores~=autoScoresBuf);
%}

clear autoScoresBuf iqr_wake iqr_NREM iqr_REM highFence_wake highFence_NREM... 
    highFence-REM lowFence_wake lowFence_NREM lowFence_REM eegRMS;

%% Write auto-scores into new TSV  %% 

disp(' ')
disp('Writing new file...')
% ------------------------------------------------------------------------
% Last edited 2018-07-20 by Yang Hu
% clarify the autoscore and data_time_epochs info and combine them

data_char = char (data);
s = nEpochs;
date_time_epoch = data_char (1:s,1:61);%make sure enough chars to cover needed sections 
time_score = string([char(date_time_epoch),num2str(autoScores,'%.6f')]);

% merge headers and data (just for keeping original format)
% ------------------------------------------------------------------------
fileID = fopen(fileName);
if rejFrac~=0
    fileOutName = [fileName(1:length(fileName)-4) ' ' num2str(round(1-rejFrac,2)) 'SemiAuto needsFill.tsv'];% Last edited 2018-07-20 by Yang Hu
else
    fileOutName = [fileName(1:length(fileName)-4) ' fullAuto.edf'];
end
fopen(fileOutName, 'w');
fileOutID = fopen(fileOutName, 'a');
fileOut_formatSpec = '%s\n';
fprintf (fileOutID, fileOut_formatSpec, headers,time_score);
fclose all;
fprintf ('New scoring file generated. \n')


%% Closing Messages

disp(' ')
disp('Input file was:')
disp(['   ' fileName])
disp(' ');
disp('Autoscored file written to:')
disp(['   ' fileOutName])
disp(' ')
toc
disp(' ')
disp(['Number of training scores: ' num2str( sum(trainSet))])
disp(['Rejection rate: ' num2str(rejFrac * 100) '%'])
disp(['Number of Unscored epochs: ' num2str( sum(autoScores == 255)) '/' num2str( nEpochs) ...
    ' (' num2str( round( sum(autoScores == 255) / nEpochs, 4) * 100) '%)'])
disp(['   Number of possible NREM/REM artifacts set to Unscored: ' num2str( nExcludeNR)])
disp(['   Number of possible Wake artifacts set to Wake-X: ' num2str( nExcludeW)])
disp(['   Number of improbable sequence epochs set to Unscored: ' num2str( nImprobable)])
disp(['   Number of training scores overriding autoscores: ' num2str( nOverride)])

% a few more messages added by Yang Hu
disp(['   Number of unscored epoch in overriding scores: ' num2str( nUnscored_in_Override)])
disp(['   Agreement rate between human scorer and autoscorer: '  num2str( 100 - (nOverride - nUnscored_in_Override ) / sum(trainSet) * 100) '%'])
disp(['   Number of empty EEG/EMG signals in each band: ' num2str( nEmpty_EEG_EMG)])
disp(['   Number of misrecoreded EEG/EMG in original file: ' num2str( misEpoch)])
disp(' ')

