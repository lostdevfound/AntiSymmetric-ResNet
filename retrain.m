% get the Network
% Reinit weight entries above the threshold with random values
% Call training but with modified weight update

function void = retrain(net)
    for i=2:net.totalNumLayers
        % Get concavity matrix
        concavW = net.ConcavW{i}
        % Init random matrix
        R = net.initScaler*normrnd(0,1,size(concavW));
        % Init mask
        M = ones(size(concavW));
        % Find weights with concavity above threshold
        [r,c] = find(concavW > threshold);

        for j=1:size(r)
            M(r(j),c(j)) = 0
        end

        M_reverse = ones(size(concavW))- M;

        % Randomize weights that are above the threshold
        net.W{i} = net.W{i} .* M + R .* M_reverse;
        net.M{i} = M_reverse;
    end
