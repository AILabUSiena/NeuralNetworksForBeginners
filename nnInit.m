function nn = nnInit(hiddenLayerSize,inputSize,outputSize)
% nnInit : initialization of the variable of a 2-layers neural network
%
%     nn = nnInit(hiddenLayerSize,inputSize,outputSize)
%
%     hiddenLayersSize : number of hidden units for the hidden layer
%     inputSize : input space dimension
%     outputSize : output space dimension

nn.wH = rand(hiddenLayerSize,inputSize)*0.5-0.2; % hidden layer weights
nn.bH = rand(hiddenLayerSize,1)*0.5-0.2; % hidden layer bias
    
nn.wO = rand(outputSize,hiddenLayerSize)*0.5-0.2; % output layer weights
nn.bO = rand(outputSize,1)*0.5-0.2; % output layer bias

nn.Loss = [];
nn.Accuracy = [];

