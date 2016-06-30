function kernel(x_1,x_0,h)
	if ((torch.abs(x_1-x_0))/h <= 0.5 ) then
		return 1;
	else
		return 0;
	end
end

function parzenPDF(input, h_0)
	-- this function returns the target constructed using the parzen window pdf estimation on a given input
	assert(input, "no data has been provided ")
	h_0 = h_0 or 16

	n = input:size(1)
	ker = torch.zeros(n):typeAs(input);
	target = torch.zeros(n):typeAs(input);
	h = h_0/torch.sqrt(n);
	for i = 1,n do
		for j = 1,n do
			if i ~= j then		
				ker[i] = ker[i] + kernel(input[i],input[j],h)
			end
		end
		target[i] = ker[i]/(n*h)
	end
	print(target)
	return target
end

function truePDF(x)
	-- normal distribution
	local x = -torch.pow(x,2)/2
	local expx = x:exp()
	return expx*(1/torch.sqrt(math.pi*2))
end




