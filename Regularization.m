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
            firstCol(1:n,1) = -2.0/n*(sum(W_prev(:,:)) - n*sum(W(:,1)) + sum(W_next(:,1)));
            otherCols = 2.0/n*(n*W - sum(W_next(:,1)));
            otherCols(:,1) = firstCol;
            dRdW = otherCols;
        end

        function dRdW = difEnd(W_prev, W_L)
            sm = @Regularization.sm;
            [n,~] = size(W_L);
            firstCol(1:n,1) = -2.0/n*(sum(W_prev(:,:)) - n*sum(W_L(:,1)));
            dRdW = [firstCol, zeros(n, n-1)];
        end


        function dRdW = difStart(W, W_next)
            sm = @Regularization.sm;
            [n,~] = size(W);
            c = (1.0/n);
            dRdW = W - c*sm(W_next);
        end


        function columnSumMatrix = sm(W)
            [numRow, numCol] = size(W);

            for i=1:numCol
                columnSumVector(i:1) = sum(W(:,i));
            end

            columnSumMatrix = repmat(columnSumVector, 1, numCol);
        end


    end
end
