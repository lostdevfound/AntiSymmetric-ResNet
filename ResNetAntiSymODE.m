% ResNetAntiSymODE class implements pure ODE like NN

classdef ResNetAntiSymODE < handle
% ResNet class with cross entropy objective function
    properties
        name;   % some str value
        tm;     % testmode, this param is used for testing gradients of the NN
        h;
        r;      % Regularization value
        hIO;    % h value for W_2 and W_YN, this var is needed for NN that was interpolated
        igamma;
        initScaler;
        numHiddenLayers;    % hiddenLayers are layers 3, 4,...,L-1
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        W2_lin;
        b2_lin;
        WYN_lin;
        bYN_lin;
        f;      % Activation function handle
        df;     % Derivative of an activation function
        ddf;    % Second derivative of an activation function
        K;      % Weights for W
        W;      % Function of K, final weights
        b;
        totalNumLayers;
        O;      % An array of omegas
        DY;     % An array of derivatives dY^(l)/dY^(l-1)
        Y;      % An array Y stores  layer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. y^(l)
        G;      % Gamme matrix
    end

    methods
        function obj = ResNetAntiSymODE(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode, activFunc, regular, p, s)
            % ResNet constructor

            ActivClass = ActivFunc(activFunc, i_testMode, p, s);
            obj.f = @ActivClass.activf;
            obj.df = @ActivClass.activfD;
            obj.ddf = @ActivClass.activfDD;
            obj.r = regular;
            obj.tm = i_testMode;
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_inputLayerSize;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = 0;
            obj.initScaler = initScaler;
            obj.igamma = i_gamma;
            obj.h = h;
            obj.hIO = h;
            % Init arrays
            obj.K{1} = 0;
            obj.D{1} = 0;   % Array of gradients dC/dY
            obj.O{1} = 0;   % Array of Omega gradients dY_i^(l)/dW_ij^(l)
            obj.Y{1} = 0;   % Array of forward pass layers Y
            obj.DY{1} = 0;  % Array of dY^(l)/dY^(l-1)
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as vector with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as vector with all enries 0

            obj.G = obj.igamma * eye(obj.hiddenLayersSize);

            % Build intermediate W and b
            for i = 2:obj.numHiddenLayers + 1
                obj.K{i} = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.hiddenLayersSize]);
                obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                obj.b{i} = obj.initScaler * normrnd(0,1,[obj.hiddenLayersSize,1]);
            end

            obj.totalNumLayers = i;
        end


        function result = forwardProp(obj, i_vector)
            % Forward propagation
            YN = obj.totalNumLayers;
            obj.Y{1} = i_vector;

            for i=2:YN
                obj.Y{i} = obj.Y{i-1} + obj.h*obj.f(obj.W{i},obj.Y{i-1},obj.b{i});
            end

            result = obj.Y{end};

        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false
            obj.Y{1} = i_vector;
            YN = obj.totalNumLayers;

            % Build softmax layer
            h_vec = ActivFunc.softmax(obj.Y{end});

            % Build dh/dY^(L) matrix, i.e deriv of softmax h w.r.t y^(L)
            dh = [];
            for i =1:obj.outputLayerSize
                for j=1:obj.outputLayerSize
                    if i==j
                        dh(i,j) = h_vec(i)*(1-h_vec(j));
                    else
                        dh(i,j) = -h_vec(i)*h_vec(j);
                    end
                end
            end

            % Calculate the last layer error gradient dC/dY^(L) (CHECKED)
            obj.D{YN} = dh' * (-label_vector ./ h_vec');
            % Calculate the last layer weights gradient dY^(L)/dW^(L)
            obj.O{YN} = obj.h * obj.df(obj.W{YN},obj.Y{YN-1}, obj.b{YN}) * obj.Y{YN-1}';

            % Calculate error gradient for L-1, L-2,..., 2 layers
            for i = YN-1:-1:2
                % Compute delta (CHECKED)
                obj.D{i} = obj.D{i+1} + obj.W{i+1}'*( obj.D{i+1} .* (obj.h*obj.df(obj.W{i+1},obj.Y{i},obj.b{i+1})) );
                % Compute omega
                obj.O{i} = obj.h * obj.df(obj.W{i},obj.Y{i-1}, obj.b{i}) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX (CHECKED)
            obj.D{1} = obj.D{2} + obj.h*obj.W{2}' * (obj.D{2} .* obj.df(obj.W{2}, i_vector, obj.b{2}));
            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases
                V = 0; dRdK = 0; dRdb = 0;
                for i = 2:YN
                    % Gradient update
                    if i==2
                        V = obj.W{3} - obj.W{2};
                        dRdb = -2*obj.r*(obj.b{3} - obj.b{2});
                    elseif i==YN
                        V = obj.W{YN} - obj.W{YN-1};
                        dRdb = -2*obj.r*(obj.b{YN} - obj.b{YN-1});
                    else
                        V = obj.W{i+1} - 2*obj.W{i} + obj.W{i-1};
                        dRdb = -2*obj.r*(obj.b{i+1} -2*obj.b{i} + obj.b{i-1});
                    end
                    dRdK = obj.r*(-V + V');
                    % dRdW = -2*obj.r*V;
                    obj.K{i} = obj.K{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})') - eta*dRdK;
                    obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                    obj.b{i} = obj.b{i} - eta* obj.h* obj.D{i} .* obj.df(obj.W{i}, obj.Y{i-1}, obj.b{i}) -eta*dRdb;
                end
            end
        end


        %  Compute derivate dY^(L)/dX
        function dYdX = computedYdX(obj, i_vector)
            % TODO need to change this cause y^(L) and y^(l2) are changed
            disp('this method needs to be changed cause y^(2) and y^(L) were modified');
            YN = obj.totalNumLayers;

            % Calculate the last layer gradient dY_i^(l)/dY_j^(l-1)
            obj.DY{YN-1} = obj.W{YN} .* (ones(obj.outputLayerSize,1) + obj.h*obj.df(obj.W{YN}, obj.Y{YN-1}, obj.b{YN}));

            % Calculate the L-1, L-2, ... , 2 layer gradients
            for i = YN-2:-1:2
                obj.DY{i} = obj.DY{i+1} + obj.DY{i+1} * (obj.W{i+1}.*obj.h*obj.df(obj.W{i+1}, obj.Y{i}, obj.b{i+1}));
            end

            obj.DY{1} = obj.DY{2} * obj.W{2} + obj.DY{2} * (obj.W{2} .* ( obj.h*obj.df(obj.W{2}, i_vector, obj.b{2})) );
            dYdX = obj.DY{1};
        end


        % Training
        function trainingRes = train(obj, trainData, trainLabel, cycles, eta)
            [vecSize, numVecs] = size(trainData);
            costAvg = 0;
            numSamples = 3000;

            for i = 1:cycles


                randInd = randi(numVecs);
                x = trainData(:, randInd);
                c = trainLabel(:, randInd);

                y = forwardProp(obj, x);

                softY = ActivFunc.softmax(y);

                backProp(obj, x, c, eta, true);

                costAvg = costAvg + norm(c - softY')^2;

                % Display stats
                if mod(i, numSamples) == 0
                    progress = 100*i / cycles;
                    classifRes=[softY',c];
                    signalY = [obj.matrixY];
                    minMaxSignalY = [min(signalY);max(signalY)];
                    costAvg = costAvg / double(numSamples);
                    disp(['average cost over ', num2str(numSamples, '%0d'),' samples: ', num2str(costAvg, '%0.3f'),' progress: ', num2str(progress)]);
                    weightNorms = obj.weightNorms();
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


        function m = matrixY(obj)
            m=[];
            for i=2:obj.totalNumLayers-1
                m(:,i) = obj.Y{i};
            end
        end


        function normsWeight = weightNorms(obj)
            for i = obj.totalNumLayers:-1:1
                normsWeight(i) = norm(obj.W{i});
            end
        end

        function arrayY = getArrayY(obj)
            for i=1:obj.totalNumLayers
                arrayY{i} = obj.Y{i};
            end
        end
    end
end
