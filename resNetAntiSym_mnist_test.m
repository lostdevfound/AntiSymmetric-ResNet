% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
igamma = 0.1;       % default 0.1
trainCycles = 500000;       % default 400000
eta = 0.002;           % good default 0.0005 or 0.003
initScaler = 0.01;      % default 0.01
neurons = 70;
layers = 5;
h = 0.4;        % default 0.7
regular = 0.1;        % regularization
% Define different params for several NNs
% numNets = 13;
% listLayers =  [2,    2,   3,   3,    5,    10,  10,  10,   10,  20,   20,   30,  30];
% listH =       [0.8, .35,  .3,  .8,   .25,  .2,  .6,  .8,   1,   .4,   .8,   .05,  0.1];
% listNeurons = [100,  100, 100, 100,  50,   20,  20,  20,   20,  20,   20,   15,  15];

% Set to true if need to retrain
first_time_launch = false;
doPerturbation = true;

% Create multiple neural nets with different params
% for i=13:numNets

    % layers = listLayers(i);
    % neurons = listNeurons(i);
    % h = listH(i);

    % Training part.
    if first_time_launch == true
        % Init NN and train it
        net = ResNetAntiSym(layers, 784, 10, neurons, igamma, h, initScaler, regular, false);
        disp('training...');
        net.train(trainImages, trainLabels, trainCycles, eta);
        disp('training complete.');

        % Save trained net
        netStr = {'antisym_net_l', num2str(layers),'_h', num2str(h),'_ig', num2str(igamma),'_n', num2str(neurons),'_r', num2str(regular),'.mat','resources/'};
        str = strcat(netStr{12},netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9}, netStr{10}, netStr{11});
        net.name = strcat(netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9}, netStr{10});
        save(str,'net');
    else
        load('resources/antisym_net_l5_h0.4_ig0.1_n70_r0.1.mat');    % Load pretrained AntiSymResNet
    end
% end


%
%              Perturbation part
%
normSum = 0;
samples = 0;
offset=43;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testImg =  validatimages(:,index);
    [~,digitNumber] = max(validatLabels(:,index));

    perturbedImg = testImg;
    classifRes = ones(10,1);

    noisyImg = min(testImg + 0.5*rand(784,1), 1);   % limit the range from 0 to 1

    % Perturbation generation
    count = 0;
    breakCount = 5000;

    while doPerturbation == true
        net.forwardProp(perturbedImg);
        perturbedImg = net.adversBackProp(perturbedImg,validatLabels(:,index), 0.01);
        classifRes = softmax(net.forwardProp(perturbedImg));

        [prediction,maxInd] = max(classifRes);

        % Break if correct classif index is not max
        if maxInd ~= digitNumber
            break;

        elseif count > breakCount
            perturbedImg = testImg;
            samples = samples -1;
            disp('breaking...........');
            break;
        end

        count = count + 1;
    end
    absNormOfPerturbation = norm(perturbedImg - testImg);
    relNormOfPerturbation = norm(perturbedImg - testImg)/norm(testImg);
    normSum = normSum + relNormOfPerturbation;
end

RelativeAvgNorm = normSum/(samples+1)

% Classify images
classificationResultPerturb = softmax(net.forwardProp(perturbedImg))';
classificationResultOrig = softmax(net.forwardProp(testImg))';
classificationResultNoisy = softmax(net.forwardProp(noisyImg))';
results = [classificationResultPerturb, classificationResultOrig, classificationResultNoisy]

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
imshow(digitOrig*255,[0 255])      % show the image
title('original');

subplot(1,3,3);
noisyImg = noisyImg + validMean;
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');



function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end


function resSoft = softmax(y_args)
    % This function computes softmax
    y_argsSum = 0;
    inputSize = max(size(y_args));

    for i = 1:inputSize
        y_argsSum = y_argsSum + exp(y_args(i));
    end

    for i = 1:inputSize
        resSoft(i) = exp(y_args(i)) / y_argsSum;
    end
end
