optim = require 'optim'
require 'cunn';

fullset = torch.load('back231.dat')
shuffle = torch.randperm(68)
shuffleset = fullset
for i=1,68 do
    shuffleset.data[i] = fullset.data[shuffle[i]]
    shuffleset.label[i] = fullset.label[shuffle[i]]
    end
fullset = shuffleset

trainset = {
    data = fullset.data[{{1,68}}]:double(),
    label = fullset.label[{{1,68}}]
}


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:float() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end


nn = require 'nn'
-- net = nn.Sequential()
-- net:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- net:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- net:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- net:add(nn.SpatialConvolution(1024, 3072, 6, 6, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialConvolution(3072, 4096, 1, 1, 1, 1))
-- net:add(nn.ReLU())
-- net:add(nn.SpatialConvolution(4096, 1000, 1, 1, 1, 1))
-- net:add(nn.View(1000))
-- net:add(nn.SpatialSoftMax())
net=torch.load('model.net')
net:remove(20)
net:add(nn.Linear(1000,500))
net:add(nn.ReLU())
net:add(nn.Linear(500,17))
net:add(nn.SpatialSoftMax())
for i=1,17 do
    net.modules[i].train = false
end
-- model = torch.load('fullmodel_back.net')
-- model = model:float()
model = net:cuda()
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.0001
trainer.maxIteration = 500
trainer:train(trainset)
model = model:float()
torch.save('overfeatmodelv2.net',model)
