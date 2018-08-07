% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
NN = 'ODE-END';        % Train 'AntiSym' or 'ResNet'
trainCycles = 150000;        % default 400000
eta = 0.0001;                 % good default 0.005
neurons = 40;
layers = 10;
initScaler = 0.01;             % default 0.01
h = 0.11;                       % default 0.1
igamma = 0.0001;               % default 0.0001
regular = 0;               % default 0.002
regular1 = 0.001;              % default 0.002
regular2 = 0;              % default 0.002
activ = 'segSig';
p = 1;
s = 1;

first_time_launch = false;
doPerturbation = true;


%                                           %
%              Training part                %
%                                           %

if first_time_launch == true

    % Init NN and train it
    if strcmp(NN, 'ODE-END')
        % load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ODE-END_segSig_net_l10_h0.11_n20_p1_s1_r0.004_r1_0.001_r2_0.mat')
        net = ResNetAntiSym_ODE_END(layers, 784, 10, neurons, igamma, h, initScaler, false, activ, p, s, regular, regular1, regular2);
    elseif strcmp(NN, 'AntiSym')
        net = ResNetAntiSym(layers, 784, 10, neurons, igamma, h, initScaler, false, activ, p, s, regular, regular1, regular2);
    elseif strcmp(NN, 'ResNet')
        net = ResNetCustom(layers, 784, 20, neurons, h, initScaler, false, activ, p, s, regular, regular1, regular2);
    else
        error('Wrong NN string');
    end

    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', NN, '_',  activ, '_net_l', num2str(net.numHiddenLayers), '_h', num2str(net.h), '_n', num2str(net.hiddenLayersSize), '_p', num2str(net.p), '_s', num2str(net.s), '_r', num2str(net.r), '_r1_', num2str(net.r1),'_r2_', num2str(net.r2), '.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');
else
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ODE-END_segSig_net_l10_h0.11_n20_p1_s1_r0.004_r1_0.001_r2_0.mat')
end



%                                           %
%              Perturbation part            %
%                                           %
normSum = 0;
samples = 0;
offset=52;
pert_eta = 0.01;
pertCycles = 10000;
perturbedImg = 0;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testImg =  validatimages(:,index);
    labelVec = validatLabels(:,index);

    % Padding for ODE END NN
    if net.outputLayerSize == 0
        paddingSize = net.hiddenLayersSize - max(size(labelVec));
        labelVec= [labelVec; zeros(paddingSize, 1)];
    end
    
    noisyImg = min(testImg + 0.3*rand(784,1), 1);   % limit the range from 0 to 1

    [~,classifIndex] = max(ActivFunc.softmax(net.forwardProp(testImg))');
    [~,labelIndex] = max(labelVec);

    if labelIndex ~= classifIndex
        samples = samples - 1;
        continue;
    end

    [peturbation, perturbedImg] = PA(net, testImg, labelVec, pert_eta, pertCycles);

    if perturbedImg == 0
        samples = samples - 1;
        continue;
    end

    relNormOfPerturbation = norm(perturbedImg - testImg)/norm(testImg);
    normSum = normSum + relNormOfPerturbation;

    if doPerturbation == false
        break
    end
end

RelativeAvgNorm = normSum/(samples+1);
disp({'RelativeAvgNorm:', num2str(RelativeAvgNorm)});


% Classify images
classificationResultPerturb = ActivFunc.softmax(net.forwardProp(perturbedImg))';
classificationResultOrig = ActivFunc.softmax(net.forwardProp(testImg))';
classificationResultNoisy = ActivFunc.softmax(net.forwardProp(noisyImg))';
% results = [classificationResultPerturb, classificationResultOrig, classificationResultNoisy];
[~,classifIndexPerturbed] = max(classificationResultPerturb);
[~,classifIndexOrig] = max(classificationResultOrig);
[~,classifIndexNoisy] = max(classificationResultNoisy);
results = [classifIndexPerturbed, classifIndexOrig, classifIndexNoisy]
% Didsplay picked image
figure;
subplot(1,3,1);
perturbedImg = perturbedImg + validMean;
digitPerturbed = reshape(perturbedImg, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');

subplot(1,3,2);
testImg = testImg + validMean;
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])             % show the image
title('original');

subplot(1,3,3);
noisyImg = noisyImg + validMean;
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');
