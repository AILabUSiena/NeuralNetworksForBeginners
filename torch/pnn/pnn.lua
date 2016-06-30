require 'torch'
require 'nn'
require 'gnuplot'

require 'datasetCreation.lua'
require 'training.lua'

local options = {}
options.nTrainExamples = 2000
options.nTestExamples = 1000
options.h_0 = 16
options.nepochs = 10
options.eta = 0.01
options.cuda = false -- ENABLE CUDA only if you have it and don't forget to install cutorch and cunn packagess


local x = torch.randn(options.nTrainExamples) -- create a tensor of values taken from a normal distribution with mean=0 and std=1
local xtest = torch.randn(options.nTestExamples,1)

local truePdf = truePDF(x)
gnuplot.plot(x,truePdf, '+')
gnuplot.title("True PDF")
gnuplot.figure()

local target = parzenPDF(x, options.h_0)
gnuplot.plot(x,target,'+')
gnuplot.title("PDF estimated with Parzen Window")
gnuplot.figure()


---- PNN creation ----
local mlp = nn.Sequential() 
inputs = 1 outputs = 1 HUs = 10 -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.ReLU())
local criterion = nn.MSECriterion()  

if options.cuda then
	require 'cutorch'
	require 'cunn'
	mlp:cuda()
	criterion:cuda()
	target = target:cuda()
	x = x:cuda()
	xtest = xtest:cuda()
	batchTraining(mlp, criterion, x, target, options.nepochs, options.eta)
else 
	onlineTraining(mlp, criterion, x, target, options.nepochs, options.eta)
end

--PNN training
--onlineTraining(mlp, criterion, x, target, options.nepochs, options.eta)


--PNN test


testOutput = mlp:forward(xtest)

gnuplot.plot(xtest:view(-1):double(),testOutput:view(-1):double(),'+')
gnuplot.title("PDF estimated with the PNN")
gnuplot.figure()
