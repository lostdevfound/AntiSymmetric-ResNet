clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');
trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;
% Load the network
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ResNet_relu_net_l10_h0.9_n20_p1_s1_r0_r1_0_r2_0.mat')

samples = 500;
passes = 1;
eta = 0.04;
cycles = 5000;
universPert = 0;
inputVec = 0;
labelVec = 0;
[vecSize, numVecs] = size(trainImages);

for pass=1:passes
    disp(['passes: ', num2str(pass)]);

    for i=1:samples
        inputVec = trainImages(:,i);
        labelVec = trainLabels(:,i);
        [~,labelIndex] = max(labelVec);
        [~,classifIndex] = max(ActivFunc.softmax(net.forwardProp(inputVec)));

        % This condition ensures that we do not compute perturbation for missclassified objects.
        if classifIndex == labelIndex

            [~,classifIndex] = max(ActivFunc.softmax(net.forwardProp(inputVec + universPert)));

            if classifIndex == labelIndex
                [perturbation, perturbedVec] = PA(net, inputVec + universPert, labelVec, eta, cycles);
                universPert = (universPert + perturbation) / (0.5*norm(universPert + perturbation));
                % universPert = (universPert + perturbation);
            else
                disp('fooled using universal perturbation.');
                disp(['progress ', num2str(i/samples*100,'%0d')]);
            end
        end
    end

end
save('resources/universal','universPert');

figure;
subplot(1,3,1);
perturbedImg = inputVec + universPert + validMean;
digitPerturbed = reshape(perturbedImg, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');

subplot(1,3,2);
testImg = inputVec + validMean;
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])             % show the image
title('original');

subplot(1,3,3);
testImg = universPert;
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])             % show the image
title('original');
