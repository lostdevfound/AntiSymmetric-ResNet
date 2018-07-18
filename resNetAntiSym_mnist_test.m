% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
igamma = 0.1;               % default 0.1
trainCycles = 400000;       % default 400000
eta = 0.002;                % good default 0.0005 or 0.003
initScaler = 0.01;          % default 0.01
neurons = 70;
layers = 5;
h = 0.4;                    % default 0.7
regular = 0.1;              % regularization
activFunc = 'relu';
p = 1;
s = 1;



% Set to true if need to retrain
first_time_launch = true;
doPerturbation = true;

% Training part.
if first_time_launch == true
    % Init NN and train it
    net = ResNetAntiSym(layers, 784, 10, neurons, igamma, h, initScaler, false, activFunc, regular, p, s);
    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', 'antisym_', activFunc, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(neurons), '_p', num2str(p), '_s', num2str(s), '_r', num2str(regular),'_gamma',num2str(igamma),'.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');

else
    load('resources/antisym_net_l5_h0.4_ig0.1_n70_r0.1.mat');    % Load pretrained AntiSymResNet
end


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
        classifRes = ActivFunc.softmax(net.forwardProp(perturbedImg));

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
