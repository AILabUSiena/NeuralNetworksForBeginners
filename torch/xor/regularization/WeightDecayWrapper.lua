local WeightDecay, parent = torch.class('nn.WeightDecayWrapper', 'nn.Sequential')

function WeightDecay:__init()
	parent.__init(self)
	self.weightDecay = 0
	self.currentOutput = 0
end

function WeightDecay:getWeightDecay(alpha)
	local alpha = alpha or 0
	local  weightDecay = 0
	for i=1,#self.modules do
		local params,_ = self.modules[i]:parameters()
		if params then
			for j=1,#params do
				weightDecay = weightDecay + torch.dot(params[j], params[j])*alpha/2
			end
		end
	end
	self.weightDecay = weightDecay
	return self.weightDecay
end

function WeightDecay:updateParameters(learningRate,alpha)
   local alpha = alpha or 0
   for i=1,#self.modules do
   	   local params, gradParams = self.modules[i]:parameters()
	   if params then
	      for j=1,#params do
	        params[j]:add(-learningRate, gradParams[j] + (alpha*params[j]))
	      end
	   end
	end
end