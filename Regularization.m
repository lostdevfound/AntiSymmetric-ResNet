classdef Regularization
    properties
        typeReg;
    end

    methods (Static)

        function dRdW = L2(W)
            dRdW = 2*W;
        end


        function dRdW = L1(W)
            dRdW = sign(W);
        end


        function dRdW = difInter(W_prev, W, W_next)
            sm = @Regularization.sm;

            [n,~] = size(W);
            c = (1.0/n);
            % dRdW = -1*(W_prev - c*sm(W))*eye(n) + n*(W - c*sm(W_next)) + (W_prev' -c*sm(W'))*(eye(n) - ones(n,n)) + n*(W' - c*sm(W_next'));
            dRdW = -c*(W_prev - c*sm(W))*eye(n) + (W -c*sm(W_next)) - c*(sm(W_prev') - sm(W));
        end


        function dRdW = difStart(W, W_next)
            sm = @Regularization.sm;
            [n,~] = size(W);
            c = (1.0/n);
            dRdW = W - c*sm(W_next);
        end


        function dRdW = difEnd(W_prev, W_L)
            sm = @Regularization.sm;
            [n,~] = size(W_L);
            c = (1.0/n);
            dRdW = -c*(W_prev - c*sm(W_L))*eye(n) - c*(sm(W_prev') - sm(W_L));
        end


        function columnSumMatrix = sm(W)
            W = W';
            [numRow, numCol] = size(W);

            for i=1:numCol
                columnSumVector(i:1) = sum(W(i,:));
            end

            columnSumMatrix = repmat(columnSumVector, 1, numCol);
        end


    end
end
