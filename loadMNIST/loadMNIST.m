% Prep training data
function [trainImages,trainLabels, validatimages, validatLabels] = loadMNIST(trainImgDir, trainLabelDir, validImgDir, validLabelDir)
    trainImages = loadMNISTImages(trainImgDir);
    trainLabels = loadMNISTLabels(trainLabelDir);
    trainLabels = trainLabels';
    trainLabels(trainLabels==0) = 10;    % replace all 0 with 10
    trainLabels = dummyvar(trainLabels)';      % make a matrix 60000x10

    % Load validation data
    validatimages = loadMNISTImages(validImgDir);
    validatLabels = loadMNISTLabels(validLabelDir);
    validatLabels = validatLabels';
    validatLabels(validatLabels==0) = 10;    % replace all 0 with 10
    validatLabels = dummyvar(validatLabels)';      % make a matrix 60000x10
end
