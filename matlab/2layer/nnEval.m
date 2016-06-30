function nn = nnEval(nn,X)
% nnEval : compute values of neurons 'a' and its activation 'z' for hidden
%          and output layers of the 2-layer Neural Network structure 'nn' 
%          on the provided input 'X'
%
%     nn = nnEval(nn,X)
%
%     nn : structure containing variables for a 2-layer Neural Network, 
%          assuming ReLu as activation and linear output 
%     X : (input_size)-by-(number_of_samples) data matrix

nn.aH = nn.wH * X + repmat(nn.bH,1,size(X,2)); % hidden connection computing

nn.zH = max(nn.aH,0); % activation

nn.o = nn.wO * nn.zH + repmat(nn.bO,1,size(X,2)); % output connection computing

end

