
function trueXorDataset()
	-- generate the classic 4 samples where True is 0.5, False -0.5 
	local dataset = torch.Tensor(4,2):fill(-1)
	local target = torch.Tensor(4,1):fill(-1)
	dataset[2][1] = 1 -- ex T F
	dataset[3][2] = 1 -- ex F T
	dataset[4][1] = 1; dataset[4][2] = 1 -- ex T T
	dataset:mul(0.5)

	target[2][1] = 1; target[3][1] = 1
	target:mul(0.5)
	return dataset, target
end

function fuzzyXorDataset(n)
	-- generate n 2D examples in [-0.5,0.5], supervising as positive the ones with the elements having
	-- the same sign and as negative otherwise
	local n = n or 1000
	local dataset = torch.rand(n,2):add(-0.5)
	local target = torch.rand(n):fill(0)

	for i=1,dataset:size(1) do
		local pattern =  dataset[i]
		if pattern[1]*pattern[2] > 0 then
			target[i] = 0.5
		else
			target[i] = -0.5
		end
	end	
	return dataset, target
end