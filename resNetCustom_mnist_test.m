% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
h = 1;                          % default 0.7
igamma = 0;                     % default 0.1
trainCycles = 400000;           % default 400000
eta = 0.002;                    % good default 0.0005 or 0.003
initScaler = 0.1;               % default 0.01
n = 20;                         %neurons
layers = 2;
activ = 'relu';
p = 5;
s = 2;
r = 0.003;

first_time_launch = false;
doPerturbation = true;
%                                           %
%              Training part                %
%                                           %

if first_time_launch == true
    % Init NN and train it
    net = ResNetCustom(layers, 784, 10, n, h, initScaler, false, activ, p, s, r);
    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/','custom_',  activ, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(n), '_p', num2str(p), '_s', num2str(s), '_r', num2str(r),'.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');
else
    load('resources/custom_relu_net_l2_h1_n20_r0.003.mat');    % Load pretrained AntiSymResNet
end



%                                           %
%              Perturbation part            %
%                                           %
normSum = 0;
samples = 1;
offset=3;
pert_eta = 0.01;
pertCycles = 10000;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testImg =  validatimages(:,index);
    labelVec = validatLabels(:,index);

    noisyImg = min(testImg + 0.3*rand(784,1), 1);   % limit the range from 0 to 1

    [peturbation, perturbedImg] = PA(net, testImg, labelVec, pert_eta, pertCycles);

    relNormOfPerturbation = norm(perturbedImg - testImg)/norm(testImg);
    normSum = normSum + relNormOfPerturbation;
end

RelativeAvgNorm = normSum/(samples+1)

% Classify images
classificationResultPerturb = ActivFunc.softmax(net.forwardProp(perturbedImg))';
classificationResultOrig = ActivFunc.softmax(net.forwardProp(testImg))';
classificationResultNoisy = ActivFunc.softmax(net.forwardProp(noisyImg))';
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
