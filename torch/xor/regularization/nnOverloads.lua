function nn.Module:updateParameters(learningRate,alpha)
   local params, gradParams = self:parameters()
   local alpha = alpha or 0
   if params then
      for i=1,#params do
        params[i]:add(-learningRate, gradParams[i] + (alpha*params[i]))
      end
   end
end

function nn.Container:updateParameters(learningRate, alpha)
    self:applyToModules(function(module) module:updateParameters(learningRate, alpha) end)
end
