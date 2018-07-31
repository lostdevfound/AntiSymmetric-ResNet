% ResNetSoftMax class implements a Residual Neural Network with custom activation function

classdef ResNetCustom < handle
% ResNet class with cross entropy objective function
    properties
        name;   % some str value
        tm;     % testmode, this param is used for testing gradients of the NN
        activFunc;
        h;
        hIO;    % h value for W_2 and W_YN, this var is needed for NN that was interpolated
        initScaler;
        numHiddenLayers;
        inputLayerSize;
        outputLayerSize;
        hiddenLayersSize;
        totalNumLayers;
        W2_lin;
        b2_lin;
        WYN_lin;
        bYN_lin;
        W;      % Weights
        b;      % Biases
        O;      % An array of omegas
        DY;     % An array of derivatives dY^(l)/dY^(l-1)
        Y;      % An array Y stores  layer vectors
        D;      % An arrat that stores Delta vectors. Delta represent the derivative of of the CostFunction w.r.t. y^(l)
        f;      % Activation function handle
        df;     % Derivative of an activation function
        ddf;
        p;      % Activation function extra param
        s;      % Activation function extra param
        r;      % Regularization parameter
        r1;
        r2;
    end

    methods
        function obj = ResNetCustom(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, h, initScaler, i_testMode, activFunc, p, s, r, r1, r2)
            % Build class of activation functions
            % Params: activFunc can be 'relu', 'sigmoid' or 'powerlog', param 'p' is a power for powerlog func
            obj.s = s; obj.p = p; obj.r1 = r1; obj.r2 = r2; obj.r = r;

            obj.activFunc = activFunc;
            ActivClass = ActivFunc(activFunc, i_testMode, p, s);
            obj.f = @ActivClass.activf;
            obj.df = @ActivClass.activfD;
            obj.ddf = @ActivClass.activfDD;
            % ResNet constructor
            obj.numHiddenLayers = i_numHiddenLayers;
            obj.inputLayerSize = i_inputLayerSize;
            obj.outputLayerSize = i_outputLayerSize;
            obj.hiddenLayersSize = i_hiddenLayersSize;
            obj.totalNumLayers = i_numHiddenLayers + 2;
            obj.initScaler = initScaler;
            obj.h = h;
            obj.hIO = h;
            % Init arrays
            obj.W{1} = 0;
            obj.D{1} = 0;   % Array of gradients dC/dY
            obj.O{1} = 0;   % Array of Omega gradients dY_i^(l)/dW_ij^(l)
            obj.Y{1} = 0;   % Array of forward pass layers Y
            obj.DY{1} = 0;  % Array of dY^(l)/dY^(l-1)
            obj.Y{2} = zeros(obj.hiddenLayersSize, 1);    % init array Y as matrix with all enries 0
            obj.D{2} = zeros(obj.hiddenLayersSize, 1);    % init array D as matrix with all enries 0

            % Build W2, b2 for connections from input layer to first hidden
            W2 = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.inputLayerSize]);
            b2 = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);
            obj.W{2} = W2;
            obj.b{2} = b2;

            obj.W2_lin = zeros(obj.hiddenLayersSize, obj.inputLayerSize);
            obj.b2_lin = zeros(obj.hiddenLayersSize,1);
            % obj.W2_lin = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.inputLayerSize]);
            % obj.b2_lin = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);

            % obj.WYN_lin = zeros(obj.outputLayerSize, obj.hiddenLayersSize);
            % obj.bYN_lin = zeros(obj.outputLayerSize, 1);
            obj.WYN_lin = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, obj.hiddenLayersSize]);
            obj.bYN_lin = obj.initScaler*normrnd(0,1,[obj.outputLayerSize,1]);


            % gammaMatrix = obj.igamma*eye(obj.hiddenLayersSize );

            % Build intermediate W and b
            for i = 3:obj.numHiddenLayers + 2   % build W^(3),...,W^(L-1)
                W = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.hiddenLayersSize]);
                b = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);
                obj.W{i} = W;
                obj.b{i} = b;
            end

            % Build W^(L)
            obj.W{i + 1} = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, obj.hiddenLayersSize]);
            obj.b{i + 1} = obj.initScaler*normrnd(0,1,[obj.outputLayerSize, 1]);
            [~, obj.totalNumLayers] = size(obj.W);
        end


        function result = forwardProp(obj, i_vector)
            % Forward propagation
            YN = obj.totalNumLayers;
            obj.Y{1} = i_vector;
            % obj.f first hidden layer
            obj.Y{2} = obj.W2_lin*obj.Y{1} + obj.b2_lin + obj.hIO*obj.f(obj.W{2},obj.Y{1},obj.b{2});

            % obj.f other consequent layers
            for i = 3:YN - 1
                obj.Y{i} = obj.Y{i-1} + obj.h*obj.f(obj.W{i},obj.Y{i-1},obj.b{i});
            end

            % obj.f last layer
            obj.Y{YN} = obj.WYN_lin*obj.Y{YN-1} + obj.bYN_lin + obj.hIO*obj.f(obj.W{YN},obj.Y{YN-1},obj.b{YN});

            result = obj.Y{end};

        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            if min(size(label_vector)) ~= 1
                error('label vector is not a vector')
            end
            if min(size(i_vector)) ~= 1
                error('i_vector is not a vector')
            end
            % preprocess inputs
            label_vector = label_vector(:);
            i_vector = i_vector(:);

            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false
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


            % Calculate the last layer error gradient dC/dY^(L)
            obj.D{YN} = dh' * (-label_vector ./ h_vec');
            % Calculate the last layer weights gradient dY^(L)/dW^(L)
            obj.O{YN} = (obj.h*obj.D{YN}.*obj.df(obj.W{YN},obj.Y{YN-1},obj.b{YN}))*obj.Y{YN-1}';
            % Calculate YN-1 layer gradient
            obj.D{YN-1} = obj.WYN_lin'*obj.D{YN} + obj.W{YN}'*(obj.D{YN}.*obj.df(obj.W{YN},obj.Y{YN-1},obj.b{YN})*obj.h);
            % Calculate the last layer weights gradient dY_(L-1)/dW_(L-1)
            obj.O{YN-1} = obj.h * obj.df(obj.W{YN-1}, obj.Y{YN-2}, obj.b{YN-1}) * obj.Y{YN-2}';

            % Calculate error gradient for L-2, L-3,..., 2 layers
            for i = YN-2:-1:2
                % Compute delta
                obj.D{i} = obj.D{i+1} + obj.W{i+1}'*( obj.D{i+1} .* (obj.h*obj.df(obj.W{i+1},obj.Y{i},obj.b{i+1})) );
                % Compute omega
                obj.O{i} = obj.h * obj.df(obj.W{i},obj.Y{i-1}, obj.b{i}) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX
            obj.D{1} = obj.W2_lin' * obj.D{2} + obj.h*obj.W{2}' * (obj.D{2} .* obj.df(obj.W{2}, i_vector, obj.b{2}));
            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases

                % First update dim reduction weights and biases
                % obj.W2_lin = obj.W2_lin - eta*(obj.D{2} *i_vector' + obj.r*obj.W2_lin) - eta*obj.r*2*obj.W2_lin;
                % obj.b2_lin = obj.b2_lin - eta*obj.D{2};

                obj.WYN_lin = obj.WYN_lin - eta*(obj.D{YN} * obj.Y{YN-1}' + obj.r*obj.WYN_lin);
                obj.bYN_lin = obj.bYN_lin - eta*obj.D{YN};

                % Update ReLu weights and biases for layer 2 and YN
                obj.W{2} = obj.W{2} - eta*(obj.h*obj.D{2}.* obj.df(obj.W{2}, i_vector, obj.b{2})* i_vector') - eta*obj.r*2*obj.W{2};
                obj.b{2} = obj.b{2} - eta* obj.h* obj.D{2} .* obj.df(obj.W{2}, i_vector,obj.b{2});

                obj.W{YN} = obj.W{YN} - eta*(obj.O{YN} + obj.r*obj.W{YN}) - eta*obj.r*2*obj.W{YN};
                obj.b{YN} = obj.b{YN} - eta*obj.h* obj.D{YN} .* obj.df(obj.W{YN}, obj.Y{YN-1}, obj.b{YN});

                % Update intermediate layers
                for i = 3:YN-1
                    obj.W{i} = obj.W{i} - eta*(diag(obj.D{i})*obj.O{i} + obj.r*obj.W{i}) - eta*obj.r*2*obj.W{i};
                    obj.b{i} = obj.b{i} - eta*obj.h* obj.D{i} .* obj.df(obj.W{i}, obj.Y{i-1}, obj.b{i});
                end

            end
        end


        %  Compute derivate dY^(L)/dX
        function dYdX = computedYdX(obj, i_vector)
            % TODO need to calculations cause y^(L) and y^(l2) are changed
            disp('this method needs to be changed cause y^(2) and y^(L) were modified')
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

                % Pad label vector. This is used of ResNetAntiSym_ODE_END child class
                paddingSize = obj.hiddenLayersSize - max(size(c));
                c = [c; zeros(paddingSize, 1)];

                y = forwardProp(obj, x);
                backProp(obj, x, c, eta, true);

                % Compute costAvg
                softY = ActivFunc.softmax(y);

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

            for i = obj.totalNumLayers:-1:1
                normsVec(i) = norm(obj.D{i});
            end
        end


        function normsWeight = weightNorms(obj)
            for i = obj.totalNumLayers:-1:1
                normsWeight(i) = norm(obj.W{i});
            end
        end


        function m = matrixY(obj)
            m=[];
            for i=2:obj.totalNumLayers-1
                m(:,i) = obj.Y{i};
            end
        end


        function arrayY = getArrayY(obj)
            for i=1:obj.totalNumLayers
                arrayY{i} = obj.Y{i};
            end
        end


    end
end
