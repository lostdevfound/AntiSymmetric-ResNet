% This file tests NN's backprop gradients.
% To obtain correct results NN's relu function must be changed to be a simple linear(identity) function
% and reluD must output 1. This is needed for symbolic gradients.

clear;
clc;
format long
h = 0.1;
igamma = 0.05;
% Build NN with params: i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, igamma, h, initScaler, testmode
% Here we set testmode=true to test gradients of the NN
% for actual training testmode must be false
net = AntiSymResNet(3, 5, 3, 3, igamma, h, 0.001, true);  %i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, igamma, h, initScaler

syms x1 x2 x3 x4 x5 b1 b2 w1;

x = [x1 x2 x3 x4 x5].';

x1 = 0.5;
x2 = 0.2;
x3 = 0.3;
x4 = 0.3;
x5 = 0.2;


W2 = net.getWeights(2);
% wn1 = W2(3,5);
% W2 = [ W2(1,:); W2(2,:); W2(3,1:4), w1 ]

W3 = net.getWeights(3);
% wn1 = W3(2,1);
% W3 = [W3(1,:); w1, W3(2,2:3); W3(3,1:3)]

W4 = net.getWeights(4);
W5 = net.getWeights(5);
wn1 = W5(3,1);
W5 = [ W5(1,:); W5(2,:);w1, W5(3,2:3) ]


b2 = net.getBias(2);
b3 = net.getBias(3);
b4 = net.getBias(4);
b5 = net.getBias(5);

z2 = W2*x + b2;
y2 = W2*x + h*relu(z2);

z3 = 0.5*(W3 - W3' - igamma*eye(3))*y2 + b3;
y3 = y2 + h*relu(z3);

z4 = 0.5*(W4 - W4' - igamma*eye(3))*y3 + b4;
y4 = y3 + h*relu(z4);

z5= W5*y4 + b5;
y5 = W5*y4 + h*relu(z5);

c = [0.1 0.5 0.2]';
C = 1/2*norm(c - sigm(y5))^2;

syms x1 x2 x3 x4 x5 b1 b2;
% dY3/dX
dYdX1 = diff(y5,x1);
dYdX2 = diff(y5,x2);
dYdX3 = diff(y5,x3);
dYdX4 = diff(y5,x4);
dYdX5 = diff(y5,x5);
% dC/dX
dCdX1 = diff(C,x1);
dCdX2 = diff(C,x2);
dCdX3 = diff(C,x3);
dCdX4 = diff(C,x4);
dCdX5 = diff(C,x5);

dCdW = diff(C,w1);


x1 = 0.5;
x2 = 0.2;
x3 = 0.3;
x4 = 0.3;
x5 = 0.2;
w1 = wn1;

% y5 = double(subs(y5))
netProp = net.forwardProp(double(subs(x)))
net_dCdx = net.backProp(double(subs(x)), c, 0.1, false)
% net_dYdX = net.computedYdX(double(subs(x)))

% dCdX = [double(subs(dCdX1)), double(subs(dCdX2)), double(subs(dCdX3)), double(subs(dCdX4)), double(subs(dCdX5))]
% dYdX = [double(subs(dYdX1)), double(subs(dYdX2)), double(subs(dYdX3)), double(subs(dYdX4)), double(subs(dYdX5))]
dCdW = double(subs(dCdW))



function y = relu(z)
% ReLu activation fucntion
    % y = max(0, z);
    y = z;
end

function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end
