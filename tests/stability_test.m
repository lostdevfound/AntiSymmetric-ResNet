clear;
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');
index = 22;

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/interpolated.mat')

x = validatimages(:,index);
net.forwardProp(x);
eulerStabilityTest(net);

function result = eulerStabilityTest(net)
    avgStability = 0;
    count = 0;

    for i=3:net.totalNumLayers-1
        jacobian = diag(net.df(net.W{i}, net.Y{i-1}, net.b{i}))*net.W{i};

        eulerStability = max(abs(1+net.h*eig(jacobian)))
        avgStability = avgStability + eulerStability;

        msg = {'euler stability, layer: ', num2str(eulerStability)};
        disp(msg);
        count = count + 1;
    end

    msg = {'avg Euler Stability', num2str(avgStability/count)};
    disp(msg);
end





% disp('############## ODE stability ##############');
% % Jacob2 = diag(ActivFunc.reluD(net.W{2}, x, net.b{2}, false))*net.W{2};
% Jacob3 = diag(net.df(net.W{3}, net.Y{2}, net.b{3}))*net.W{3};
% Jacob4 = diag(net.df(net.W{4}, net.Y{3}, net.b{4}))*net.W{4};
% Jacob5 = diag(net.df(net.W{5}, net.Y{4}, net.b{5}))*net.W{5};
% Jacob6 = diag(net.df(net.W{6}, net.Y{5}, net.b{6}))*net.W{6};
% Jacob7 = diag(net.df(net.W{7}, net.Y{6}, net.b{7}))*net.W{7};
% Jacob8 = diag(net.df(net.W{8}, net.Y{7}, net.b{8}))*net.W{8};
% Jacob9 = diag(net.df(net.W{9}, net.Y{8}, net.b{9}))*net.W{9};
% Jacob10 = diag(net.df(net.W{10}, net.Y{9}, net.b{10}))*net.W{10};
%
% % eigVals2 = max(real(eig(Jacob2)))
% eigVals3 = max(real(eig(Jacob3)))
% eigVals4 = max(real(eig(Jacob4)))
% eigVals5 = max(real(eig(Jacob5)))
% eigVals6 = max(real(eig(Jacob6)))
%
% % eigval2 = max(real(eig(net.W{2})))
% eigval3 = max(real(eig(net.W{3})))
% eigval4 = max(real(eig(net.W{4})))
% eigval5 = max(real(eig(net.W{5})))
% eigval6 = max(real(eig(net.W{6})))
% eigval7 = max(real(eig(net.W{7})))
% eigval8 = max(real(eig(net.W{8})))
%
% disp('############## Forward Euler Method Lemma1 ##############');
% % eulerStability2 = max(abs(1+net.h*eig(Jacob2)))
% eulerStability3 = max(abs(1+net.h*eig(Jacob3)))
% eulerStability4 = max(abs(1+net.h*eig(Jacob4)))
% eulerStability5 = max(abs(1+net.h*eig(Jacob5)))
% eulerStability6 = max(abs(1+net.h*eig(Jacob6)))
% eulerStability7 = max(abs(1+net.h*eig(Jacob7)))
% eulerStability8 = max(abs(1+net.h*eig(Jacob8)))
% eulerStability9 = max(abs(1+net.h*eig(Jacob9)))
% eulerStability10 = max(abs(1+net.h*eig(Jacob10)))
