function onlineTraining(mlp, criterion, y, target, nepochs, eta)
	local nepochs = nepochs or 10 -- how to assign default values to function parameters
	local eta = eta or 0.01 -- how to assign default values to function parameters		

	local input = torch.Tensor(1);
	local output = torch.Tensor(1);

	for e = 1,nepochs do
		for k = 1,n do
			input[1] = y[k]
			output[1] = target[k]
		  	criterion:forward(mlp:forward(input), output)
			mlp:zeroGradParameters()
			-- (2) accumulate gradients
			mlp:backward(input, criterion:backward(mlp.output, output))
			-- (3) update parameters with a 0.01 learning rate
			mlp:updateParameters(eta)
		end
	end
end

function batchTraining(mlp, criterion, input, target, nepochs, eta)
	local nepochs = nepochs or 10 -- how to assign default values to function parameters
	local eta = eta or 0.01 -- how to assign default values to function parameters		

	local input = input:view(-1,1)
	local target = target:view(-1,1)
	print(input:size())
	print(target:size())
	for e = 1,nepochs do
		for k = 1,n do			
		  	criterion:forward(mlp:forward(input), target)
			mlp:zeroGradParameters()
			-- (2) accumulate gradients
			mlp:backward(input, criterion:backward(mlp.output, target))
			-- (3) update parameters with a 0.01 learning rate
			mlp:updateParameters(eta)
		end
	end
end
