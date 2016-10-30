require "nn"
require "gnuplot"
require './regularization/includes.lua'
require './data/includes.lua'
require './plots/includes.lua' -- include some routines to plot some results
---- Neural Network Creation ----
--mlp = nn.Sequential();  -- make a multi-layer perceptron
mlp = nn.WeightDecayWrapper()
inputs = 2; outputs = 1; HUs = 2; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

---- Loss Function ----
criterion = nn.MSECriterion()

local dataset, target = trueXorDataset()
--local dataset, target = fuzzyXorDataset()

print('Dataset');print(dataset)
print('Target');print( target)

---- Training the Network ----
nepochs = 1000; learning_rate = 0.05;
alpha = 0 -- weight decay coefficent (0 means no weight decay contribute)
local loss = torch.Tensor(nepochs):fill(0); local weightDecay = torch.Tensor(nepochs):fill(0)
for i = 1,nepochs do
  local input = dataset
  loss[i] = criterion:forward(mlp:forward(input), target) -- feed the net and the criterion
  weightDecay[i] = mlp:getWeightDecay(alpha)

  mlp:zeroGradParameters()   -- zero the accumulation of the gradients
  mlp:backward(input, criterion:backward(mlp.output, target))   -- accumulate gradients
  mlp:updateParameters(learning_rate, alpha)   -- update parameters with a learining rate learning_rate and weight decay coefficent of alpha
end

----  Test the Network ----
plotSepSurface(mlp)
plotLoss(loss, weightDecay)
plotPredictions(mlp)
