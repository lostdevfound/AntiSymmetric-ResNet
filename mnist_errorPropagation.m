clear all;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

NN ='AntiSym'

if strcmp(NN,'AntiSym')
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ODE-END_segSig_net_l10_h0.11_n20_p1_s1_r0.004_r1_0.001_r2_0.mat')
elseif strcmp(NN,'ResNet')
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ResNet_segSig_net_l10_h0.11_n20_p1_s1_r0.005_r1_0.001_r2_0.mat')
else
    eror('Wrong NN str')
end

index = 99;
testVec = validatimages(:,index);
labelVec = validatLabels(:,index);


eta = 0.01; cycles = 10000;

% Padding for ODE end NN
paddingSize = net.hiddenLayersSize - max(size(labelVec));
labelVec= [labelVec; zeros(paddingSize, 1)];

[perturbation, perturbedVec] = PA(net, testVec, labelVec, eta, cycles);

disp({'perturbation norm:', num2str(norm(perturbedVec-testVec,2))});

classifOriginal = net.forwardProp(testVec);
correctPropagation = net.getArrayY();

classifPerturbed = net.forwardProp(perturbedVec);
fooledPropagation = net.getArrayY();
[~,classifPertLabel] = max(ActivFunc.softmax(classifPerturbed));
[~, classifOriginalLabel] = max(ActivFunc.softmax(classifOriginal));
results = [classifPertLabel, classifOriginalLabel];
disp('Classification Results: Perturbed Original')
disp(results);

differenceNorms = [];

% Propagation from y^2 to y^L
for i=1:net.totalNumLayers
    differenceNorms(i) = (norm(correctPropagation{i} - fooledPropagation{i}));
end

disp('difference norms')
differenceNorms

disp('index:');disp(index);

figure
semilogy(differenceNorms)
legend('norm of error')
title({NN, 'log scale'})
xlabel('layers')
ylabel('norm of error')

figure;
subplot(1,2,1);
perturbedVec = perturbedVec + validMean;
digitPerturbed = reshape(perturbedVec, [28,28]);    % row = 28 x 28 image
% imshow(digitPerturbed*255, [0 255])      % show the image
imshow(digitPerturbed)      % show the image
title('perturbed');

subplot(1,2,2);
testVec = testVec + validMean;
digitOrig = reshape(testVec, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])             % show the image
title('original');
