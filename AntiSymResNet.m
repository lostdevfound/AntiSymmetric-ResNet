% Antisymmetric ResNet class

classdef AntiSymResNet < handle
% Anti-Symmetric ResNet class
    properties
        tm;     % testmode, this param is used for testing gradients of the NN
        h;
        igamma;
        initScaler;
        numHiddenLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        arrayWeights;
        arrayBiases;
        totalNumLayers;
        O;      % An array of omegas
        DY;     % An array of derivatives dY^(l)/dY^(l-1)
        Y;      % An array Y stores  layer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. y^(l)
    end

    methods
        function obj = AntiSymResNet(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, igamma, h, initScaler, i_testMode)
            % AntiSymResNet constructor
            obj.tm = i_testMode;
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_inputLayerSize;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = i_numHiddenLayers + 2;
            obj.initScaler = initScaler;
            obj.h = h;

            % Init arrays
            obj.D{1} = 0;   % Array of gradients dC/dY
            obj.O{1} = 0;   % Array of Omega gradients dY_i^(l)/dW_ij^(l)
            obj.Y{1} = 0;   % Array of forward pass layers Y
            obj.DY{1} = 0;  % Array of dY^(l)/dY^(l-1)
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = obj.initScaler*rand(obj.hiddenLayersSize, obj.inputLayerSize);
            b2 = obj.initScaler*rand(obj.hiddenLayersSize,1);

            obj.arrayWeights{2} = W2;
            obj.arrayBiases{2} = b2;

            gammaMatrix = igamma*eye(obj.hiddenLayersSize );

            % Build intermediate W and b
            for i = 3:obj.totalNumLayers - 1   % do not build W and b from last hidden layer to output layer
                K = obj.initScaler*rand(obj.hiddenLayersSize, obj.hiddenLayersSize);
                b = obj.initScaler*rand(obj.hiddenLayersSize,1);
                % make W anti-symmetric and add gamma diffusion by substracting gammaMatrix
                W = 0.5*(K - K'- gammaMatrix);
                obj.arrayWeights{i} = W;
                obj.arrayBiases{i} = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = obj.initScaler*rand(obj.outputLayerSize, obj.hiddenLayersSize);
            bN = obj.initScaler*rand(obj.outputLayerSize, 1);
            obj.arrayWeights{obj.totalNumLayers} = WN;
            obj.arrayBiases{obj.totalNumLayers} = bN;

        end


        function result = forwardProp(obj, i_vector)
            % Forward propagation
            YN = obj.totalNumLayers;
            % relu first hidden layer
            obj.Y{2} = obj.arrayWeights{2}*i_vector + obj.h*relu(obj.arrayWeights{2},i_vector,obj.arrayBiases{2}, obj.tm);
            % obj.Y{2} = i_vector + obj.h*relu(obj.arrayWeights{2},i_vector,obj.arrayBiases{2});

            % relu other consequent layers plus output layer
            for i = 3:YN - 1
                obj.Y{i} = obj.Y{i-1} + obj.h*relu(obj.arrayWeights{i},obj.Y{i-1},obj.arrayBiases{i}, obj.tm);
            end

            obj.Y{YN} = obj.arrayWeights{YN}*obj.Y{YN-1} + obj.h*relu(obj.arrayWeights{YN},obj.Y{YN-1},obj.arrayBiases{YN}, obj.tm);

            % TODO the end layer may be wrapt in the sigmoid function for 0-1 output range
            result = obj.Y{end};

        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false

            YN = obj.totalNumLayers;

            % Calculate the last layer error gradient dC/dY^(l)
            obj.D{YN} = (sigm(obj.Y{YN}) - label_vector) .* sigmD(obj.Y{YN});

            % Calculate the last layer weights gradient dY_i^(l)/dW_ij^(l)
            obj.O{YN} = ones(obj.outputLayerSize,1)*obj.Y{YN-1}' + obj.h*reluD(obj.arrayWeights{YN},obj.Y{YN-1},obj.arrayBiases{YN},obj.tm)*obj.Y{YN-1}';
            % Calculate YN-1 layer gradient
            obj.D{YN-1} = obj.arrayWeights{YN}' * (obj.D{YN}.*(ones(obj.outputLayerSize,1) + obj.h*reluD(obj.arrayWeights{YN},obj.Y{YN-1},obj.arrayBiases{YN},obj.tm)));
            obj.O{YN-1} = obj.h * reluD(obj.arrayWeights{YN-1},obj.Y{YN-2},obj.arrayBiases{YN-1},obj.tm) * obj.Y{YN-2}';
            % Calculate error gradient for L-2, L-3,..., 2 layers
            for i = YN-2:-1:2
                % Compute delta
                obj.D{i} = obj.D{i+1} + obj.arrayWeights{i+1}'*( obj.D{i+1} .* (obj.h*reluD(obj.arrayWeights{i+1},obj.Y{i},obj.arrayBiases{i+1},obj.tm)) );
                % Compute omega
                obj.O{i} = obj.h * reluD(obj.arrayWeights{i},obj.Y{i-1},obj.arrayBiases{i},obj.tm) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX
            obj.D{1} = obj.arrayWeights{2}' * (obj.D{2} .* (ones(obj.hiddenLayersSize,1) + obj.h*reluD(obj.arrayWeights{2},i_vector,obj.arrayBiases{2},obj.tm)));

            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases
                obj.arrayWeights{2} = obj.arrayWeights{2} - eta * diag(obj.D{2})*obj.O{2};
                obj.arrayBiases{2} = obj.arrayBiases{2} - eta* obj.h* obj.D{2} .* reluD(obj.arrayWeights{2},i_vector,obj.arrayBiases{2},obj.tm);

                for i = 3:YN-1
                    obj.arrayWeights{i} = obj.arrayWeights{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})');
                    obj.arrayBiases{i} = obj.arrayBiases{i} - eta* obj.h* obj.D{i} .* reluD(obj.arrayWeights{i},obj.Y{i-1},obj.arrayBiases{i},obj.tm);
                end
                obj.arrayWeights{YN} = obj.arrayWeights{YN} - eta * diag(obj.D{YN})*obj.O{YN};
                obj.arrayBiases{YN} = obj.arrayBiases{YN} - eta* obj.h* obj.D{YN} .* reluD(obj.arrayWeights{YN},obj.Y{YN-1},obj.arrayBiases{YN},obj.tm);

            end
        end


        %  Compute derivate dY^(L)/dX
        function dYdX = computedYdX(obj, i_vector)
            YN = obj.totalNumLayers;

            % Calculate the last layer gradient dY_i^(l)/dY_j^(l-1)
            obj.DY{YN-1} = obj.arrayWeights{YN} .* (ones(obj.outputLayerSize,1) + obj.h*reluD(obj.arrayWeights{YN},obj.Y{YN-1},obj.arrayBiases{YN},obj.tm));

            % Calculate the L-1, L-2, ... , 2 layer gradients
            for i = YN-2:-1:2
                obj.DY{i} = obj.DY{i+1} + obj.DY{i+1} * (obj.arrayWeights{i+1}.*obj.h*reluD(obj.arrayWeights{i+1},obj.Y{i},obj.arrayBiases{i+1},obj.tm));
            end

            obj.DY{1} = obj.DY{2} * obj.arrayWeights{2} + obj.DY{2} * (obj.arrayWeights{2} .* ( obj.h*reluD(obj.arrayWeights{2}, i_vector, obj.arrayBiases{2},obj.tm)) );
            dYdX = obj.DY{1};
        end

        % Adversarial back prop
        function perturbedVector = adversBackProp(obj, i_vector,label_vector,eta)
            % Gradient w.r.t the i_vector
            backProp(obj, i_vector, label_vector, eta, false);
            perturbator = eta*obj.D{1};
            perturbedVector = i_vector + perturbator;
        end


        % Training
        function trainingRes = train(obj, trainData, trainLabel, cycles, eta)
            [vecSize, numVecs] = size(trainData);

            for i = 1:cycles
                randInd = randi(numVecs);
                x = trainData(:, randInd);
                y = trainLabel(:, randInd);
                forwardProp(obj, x);
                backProp(obj, x, y, eta, true);
            end

        end

        function delta = getDelta(obj, id)
            delta = obj.D{id};
        end

        function weights = getWeights(obj, id)
            weights = obj.arrayWeights(id);
            weights = weights{1};
        end

        function bias = getBias(obj, id)
            bias = obj.arrayBiases(id);
            bias = bias{1};
        end
    end
end


function y = relu(W,x,b, testmode)
% ReLu activation fucntion

    if testmode == true
        y = W*x+b;    % this is for testing without max() operator
    else
        y = max(0, W*x+b);
    end
end

function d = reluD(W,x,b,testmode)
    % reluD derivative of the ReLu function
    if testmode == true
        d = 1;    % this is for testing without max() operator
    else
        d = W*x+b > 0;
    end
end

function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end

function d = sigmD(x)
    % sigmD is a derivative of the sigmoid function
    d = sigm(x) .* (1 - sigm(x));
end
