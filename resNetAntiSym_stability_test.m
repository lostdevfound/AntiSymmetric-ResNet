clc;clear;
% [trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');
% index = 33;
%
% trainMean = mean(trainImages(:,1:end),2);
% validMean = mean(validatimages(:,1:end),2);
% trainImages = trainImages - trainMean;
% validatimages = validatimages - validMean;

x =  [0.1; 0.3; 0.4];
inputNorm = norm(x)
load('resources/ODE_tan_h_net_l25_h0.3_n3_p1_s1_r0.0001_gamma0.0001.mat');

disp('############## ODE stability ##############');
Jacob2 = diag(ActivFunc.reluD(net.W{2}, x, net.b{2}, false))*net.W{2};
Jacob3 = diag(ActivFunc.reluD(net.W{3}, net.Y{2}, net.b{3}, false))*net.W{3};
Jacob4 = diag(ActivFunc.reluD(net.W{4}, net.Y{3}, net.b{4}, false))*net.W{4};
Jacob5 = diag(ActivFunc.reluD(net.W{5}, net.Y{4}, net.b{5}, false))*net.W{5};
Jacob6 = diag(ActivFunc.reluD(net.W{6}, net.Y{5}, net.b{6}, false))*net.W{6};
Jacob7 = diag(ActivFunc.reluD(net.W{7}, net.Y{6}, net.b{7}, false))*net.W{7};
Jacob8 = diag(ActivFunc.reluD(net.W{8}, net.Y{7}, net.b{8}, false))*net.W{8};
Jacob9 = diag(ActivFunc.reluD(net.W{9}, net.Y{8}, net.b{9}, false))*net.W{9};

% eigVals2 = max(real(eig(Jacob2)))
% eigVals3 = max(real(eig(Jacob3)))
% eigVals4 = max(real(eig(Jacob4)))
% eigVals5 = max(real(eig(Jacob5)))
% eigVals6 = max(real(eig(Jacob6)))

eigval2 = max(real(eig(net.W{2})))
eigval3 = max(real(eig(net.W{3})))
eigval4 = max(real(eig(net.W{4})))
eigval5 = max(real(eig(net.W{5})))
eigval6 = max(real(eig(net.W{6})))
eigval6 = max(real(eig(net.W{7})))
eigval6 = max(real(eig(net.W{8})))

disp('############## Forward Euler Method Lemma1 ##############');
eulerStability2 = max(abs(1+net.h*eig(Jacob2)))
eulerStability3 = max(abs(1+net.h*eig(Jacob3)))
eulerStability4 = max(abs(1+net.h*eig(Jacob4)))
eulerStability5 = max(abs(1+net.h*eig(Jacob5)))
eulerStability6 = max(abs(1+net.h*eig(Jacob6)))
eulerStability7 = max(abs(1+net.h*eig(Jacob7)))
eulerStability8 = max(abs(1+net.h*eig(Jacob8)))
eulerStability9 = max(abs(1+net.h*eig(Jacob9)))
