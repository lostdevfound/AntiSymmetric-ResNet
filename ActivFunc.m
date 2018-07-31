classdef ActivFunc < handle
    properties
        activ;
        testmode;
        p;
        s;
    end

    methods

        function obj = ActivFunc(activ, testmode,p ,s);
            obj.activ = activ;
            obj.testmode = testmode;
            obj.p = p;
            obj.s = s;
        end

        function y = activf(obj, W, x, b)
            % This function picks desired activation function based on th class property "activ"
            y=0;
            if strcmp(obj.activ, 'relu')
                y = ActivFunc.relu(W, x, b, obj.testmode);

            elseif strcmp(obj.activ,'sigmoid')
                y = ActivFunc.v_sigm(W, x, b);

            elseif strcmp(obj.activ, 'tan_h')
                y = ActivFunc.tan_h(W, x, b);

            elseif strcmp(obj.activ, 'segSig')
                y = ActivFunc.segSig(W, x, b);
            end
        end


        function d = activfD(obj, W, x, b)
            % This function picks desired activation function based on the string parameter "activ"
            d = 0;
            if strcmp(obj.activ, 'relu')
                d = ActivFunc.reluD(W, x, b, obj.testmode);

            elseif strcmp(obj.activ,'sigmoid')
                d = ActivFunc.v_sigmD(W, x, b);

            elseif strcmp(obj.activ, 'tan_h')
                d = ActivFunc.tan_hD(W, x, b);

            elseif strcmp(obj.activ, 'segSig')
                d = ActivFunc.segSigD(W, x, b);
            end
        end

    end


    methods (Static)

        function y = relu(W, x, b, testmode)
        % vector ReLu activation fucntion
            [~,n] = size(W);
            z = W*x+b;

            if testmode == true
                y = z;    % this is for testing without max() operator
            else
                y = max(0, z);
            end
        end


        function d = reluD(W, x, b, testmode)
            % vector ReluD derivative of the ReLu function
            leak = 0;

            if testmode == true
                d = 1;    % this is for testing without max() operator
            else
                d = W*x + b >= 0;
                d(d==0) = leak;
            end
        end


        function y = v_sigm(W, x, b)
            % vector sigmoid activation function.
            y = 1./(1+exp(-(W*x+b)));
        end


        function d = v_sigmD(W, x, b)
            % vector sigmD is a derivative of the sigmoid function
            sigm = ActivFunc.v_sigm(W,x,b);
            d = sigm .* (1 - sigm);
        end

        function y = sigm(z)
            % scalar sigmoid activation function.
            y = 1./(1+exp(-z));
        end


        function d = sigmD(x)
            % scalar sigmD is a derivative of the sigmoid function
            sigm = ActivFunc.sigm(x);
            d = sigm * (1 - sigm);
        end


        function y = tan_h(W, x, b)
            % vector hyperbolic tangent activation function.
            y = tanh(W*x+b);
        end

        function y = tan_hD(W, x, b)
            % derivative of vector hyperbolic tangent activation function.
            y = sech(W*x+b).^2;
        end


        function y = segSig(W, x, b)
            z = W*x + b;
            y = min(1, max(-1,z));
        end


        function d = segSigD(W, x, b)
            z = W*x + b;
            d = z > -1 & z < 1;
        end

        function resSoft = softmax(y_args)
            % This function computes softmax
            if min(size(y_args)) ~= 1
                error('input is not a vector in softmax')
            end
            C = max(y_args);
            resSoft = exp(y_args - C) / (sum(exp(y_args -C)));
            resSoft = resSoft';

        end
    end
end
