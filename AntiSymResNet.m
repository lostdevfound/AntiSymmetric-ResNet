% Antisymmetric ResNet class

classdef AntiSymResNet < handle
% Anti-Symmetric ResNet class
    properties
        name;   % some str value
        tm;     % testmode, this param is used for testing gradients of the NN
        h;
        hIO;    % h value for W_2 and W_YN, this var is needed for NN that was interpolated
        igamma;
        initScaler;
        numHiddenLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        W2_lin;
        WYN_lin;
        W;
        b;
        totalNumLayers;
        O;      % An array of omegas
        DY;     % An array of derivatives dY^(l)/dY^(l-1)
        Y;      % An array Y stores  layer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. y^(l)
    end

    methods
        function obj = AntiSymResNet(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode)
            % AntiSymResNet constructor
            obj.tm = i_testMode;
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_inputLayerSize;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = i_numHiddenLayers + 2;
            obj.initScaler = initScaler;
            obj.h = h;
            obj.hIO = h;
            obj.igamma = i_gamma;
            % Init arrays
            obj.D{1} = 0;   % Array of gradients dC/dY
            obj.O{1} = 0;   % Array of Omega gradients dY_i^(l)/dW_ij^(l)
            obj.Y{1} = 0;   % Array of forward pass layers Y
            obj.DY{1} = 0;  % Array of dY^(l)/dY^(l-1)
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.inputLayerSize]);
            b2 = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);

            obj.W2_lin = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.inputLayerSize]);
            obj.WYN_lin = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, obj.hiddenLayersSize]);

            obj.W{2} = W2;
            obj.b{2} = b2;

            gammaMatrix = obj.igamma*eye(obj.hiddenLayersSize );

            % Build intermediate W and b
            for i = 3:obj.totalNumLayers - 1   % do not build W and b from last hidden layer to output layer
                K = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.hiddenLayersSize]);
                b = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);
                % make W anti-symmetric and add igamma diffusion by substracting gammaMatrix
                W = 0.5*(K - K'- gammaMatrix);
                obj.W{i} = W;
                obj.b{i} = b;
            end

            % Build W and b from last hidden layer to output layer
            WN = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, obj.hiddenLayersSize]);
            bN = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, 1]);
            obj.W{obj.totalNumLayers} = WN;
            obj.b{obj.totalNumLayers} = bN;

        end


        function result = forwardProp(obj, i_vector)
            % Forward propagation
            YN = obj.totalNumLayers;
            % relu first hidden layer
            obj.Y{2} = obj.W2_lin*i_vector + obj.hIO*relu(obj.W{2},i_vector,obj.b{2},obj.igamma, obj.tm);

            % relu other consequent layers
            for i = 3:YN - 1
                obj.Y{i} = obj.Y{i-1} + obj.h*relu(obj.W{i},obj.Y{i-1},obj.b{i},obj.igamma, obj.tm);
            end

            % relu last layer
            obj.Y{YN} = obj.WYN_lin*obj.Y{YN-1} + obj.hIO*relu(obj.W{YN},obj.Y{YN-1},obj.b{YN},obj.igamma, obj.tm);

            result = obj.Y{end};

        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false

            YN = obj.totalNumLayers;

            % Calculate the last layer error gradient dC/dY^(l)
            obj.D{YN} = (sigm(obj.Y{YN}) - label_vector) .* sigmD(obj.Y{YN});

            % Calculate the last layer weights gradient dY_i^(l)/dW_ij^(l)
            obj.O{YN} = (obj.h*obj.D{YN}.*reluD(obj.W{YN},obj.Y{YN-1},obj.b{YN}, obj.igamma,obj.tm))*obj.Y{YN-1}';
            % Calculate YN-1 layer gradient
            obj.D{YN-1} = obj.WYN_lin'*obj.D{YN} + obj.W{YN}'*(obj.D{YN}.*reluD(obj.W{YN},obj.Y{YN-1},obj.b{YN},obj.igamma,obj.tm)*obj.h);
            % Calculate the last layer weights gradient dY_(l-1)/dW_(l-1)
            obj.O{YN-1} = obj.h * reluD(obj.W{YN-1}, obj.Y{YN-2}, obj.b{YN-1}, obj.igamma, obj.tm) * obj.Y{YN-2}';

            % Calculate error gradient for L-2, L-3,..., 2 layers
            for i = YN-2:-1:2
                % Compute delta
                obj.D{i} = obj.D{i+1} + obj.W{i+1}'*( obj.D{i+1} .* (obj.h*reluD(obj.W{i+1},obj.Y{i},obj.b{i+1},obj.igamma,obj.tm)) );
                % Compute omega
                obj.O{i} = obj.h * reluD(obj.W{i},obj.Y{i-1}, obj.b{i},obj.igamma, obj.tm) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX
            % TODO fix this cause weights matrices have been changed
            obj.D{1} = obj.W{2}' * (obj.D{2} .* (ones(obj.hiddenLayersSize,1) + obj.h*reluD(obj.W{2},i_vector,obj.b{2}, obj.igamma, obj.tm)));

            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases
                dCdW2_lin = obj.D{2} *i_vector';
                obj.W2_lin = obj.W2_lin - eta*dCdW2_lin;

                dCdWYN_lin = obj.D{YN} * obj.Y{YN-1}';
                obj.WYN_lin = obj.WYN_lin - eta*dCdWYN_lin;

                dCdW_2 = (obj.h*obj.D{2}.* reluD(obj.W{2}, i_vector, obj.b{2},obj.igamma, obj.tm))* i_vector';
                obj.W{2} = obj.W{2} - eta * dCdW_2;
                obj.b{2} = obj.b{2} - eta* obj.h* obj.D{2} .* reluD(obj.W{2}, i_vector,obj.b{2}, obj.igamma, obj.tm);

                for i = 3:YN-1
                    obj.W{i} = obj.W{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})');
                    obj.b{i} = obj.b{i} - eta* obj.h* obj.D{i} .* reluD(obj.W{i}, obj.Y{i-1}, obj.b{i}, obj.igamma, obj.tm);
                end


                obj.W{YN} = obj.W{YN} - eta * obj.O{YN};
                obj.b{YN} = obj.b{YN} - eta* obj.h* obj.D{YN} .* reluD(obj.W{YN}, obj.Y{YN-1}, obj.b{YN}, obj.igamma, obj.tm);
            end
        end


        %  Compute derivate dY^(L)/dX
        function dYdX = computedYdX(obj, i_vector)
            YN = obj.totalNumLayers;

            % Calculate the last layer gradient dY_i^(l)/dY_j^(l-1)
            obj.DY{YN-1} = obj.W{YN} .* (ones(obj.outputLayerSize,1) + obj.h*reluD(obj.W{YN}, obj.Y{YN-1}, obj.b{YN}, obj.igamma, obj.tm));

            % Calculate the L-1, L-2, ... , 2 layer gradients
            for i = YN-2:-1:2
                obj.DY{i} = obj.DY{i+1} + obj.DY{i+1} * (obj.W{i+1}.*obj.h*reluD(obj.W{i+1}, obj.Y{i}, obj.b{i+1}, obj.igamm, obj.tm));
            end

            obj.DY{1} = obj.DY{2} * obj.W{2} + obj.DY{2} * (obj.W{2} .* ( obj.h*reluD(obj.W{2}, i_vector, obj.b{2}, obj.igamm, obj.tm)) );
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
            costAvg = 0;
            numSamples = 3000;

            for i = 1:cycles

                progress = 100*i / cycles;

                randInd = randi(numVecs);
                x = trainData(:, randInd);
                c = trainLabel(:, randInd);
                y = sigm(forwardProp(obj, x));     % wrap into sigmoid function for 0-1 range
                backProp(obj, x, c, eta, true);

                costAvg = costAvg + norm(c - y)^2;

                if mod(i, numSamples) == 0
                    [sigm(forwardProp(obj, x)),c]     % wrap into sigmoid function for 0-1 range
                    costAvg = costAvg / double(numSamples);
                    disp(['average cost over ', num2str(numSamples, '%0d'),' samples: ', num2str(costAvg, '%0.3f'),' progress: ', num2str(progress)]);
                    gradNorms = obj.gradientNorms()
                    costAvg = 0;
                end

            end

        end

        function delta = getDelta(obj, id)
            delta = obj.D{id};
        end

        function weights = getWeights(obj, id)
            weights = obj.W(id);
            weights = weights{1};
        end

        function bias = getBias(obj, id)
            bias = obj.b(id);
            bias = bias{1};
        end

        function normsVec = gradientNorms(obj)

            for i = obj.totalNumLayers - 1:-1:2
                normsVec(i) = norm(obj.D{i});
            end
        end
    end
end


function y = relu(W, x, b, g, testmode)
% ReLu activation fucntion
    [~,n] = size(W);

    if testmode == true
        y = W*x+b;    % this is for testing without max() operator
    else
        y = max(0, W*x+b);
    end
end

function d = reluD(W, x, b, g, testmode)
    % reluD derivative of the ReLu function
    leak = 0.01;

    if testmode == true
        d = 1;    % this is for testing without max() operator
    else
        d = W*x + b >= 0;
        d(d==0) = leak;
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
