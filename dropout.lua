optim = require 'optim'
require 'cunn';

fullset = torch.load('data/size_100.dat') -- Shuffling fullset
size = fullset.size
shuffle = torch.randperm(size)
shuffleset = fullset
for i=1,size do
    shuffleset.data[i] = fullset.data[shuffle[i]]
    shuffleset.label[i] = fullset.label[shuffle[i]]
    end
fullset = shuffleset

-- Trainset 80% of fullset
trainset = {
    data = fullset.data[{{1,size*.8}}]:float(),
    label = fullset.label[{{1,size*.8}}],
    size = size*.8
}

-- Testset 20% of fullset
testset = {
    data = fullset.data[{{size*.8+1,size}}]:float(),
    label = fullset.label[{{size*.8+1,size}}],
    size = size-size*.8
}

-- Converting trainset to meatable
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);
trainset.data = trainset.data:float() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end

-- Model
nn = require 'nn'
model = nn.Sequential()
model:add(nn.SpatialConvolution(3,6,5,5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(6,16,5,5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(16,20,5,5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.View(20*9*9))
model:add(nn.Dropout(.2))
model:add(nn.Linear(20*9*9,100))
model:add(nn.ReLU())
model:add(nn.Linear(100,17))
model:add(nn.LogSoftMax())

-- model = torch.load('modelv1.net')  -- loading the model
model = model:cuda()  -- model for gpu
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainset.data = trainset.data:cuda()    -- Trainset for cuda
trainset.label = trainset.label:cuda()

trainer = nn.StochasticGradient(model, criterion) --Training hyperparameters
trainer.learningRate = 0.0001
trainer.learningRateDecay = 0.09
trainer.maxIteration = 1000
trainer:train(trainset)


testset.data = testset.data:cuda()  -- Testset for cuda
testset.label = testset.label:cuda()

eval = function(dataset)      -- evalutation of testset
    correct = 0
    for i=1,dataset.size do
        local target = dataset.label[i]
        local prediction = model:forward(dataset.data[i])
        local confidences, indices = torch.sort(prediction, true)
        if target == indices[1] then
            correct = correct + 1
        end
    end
    return correct/dataset.size*100
end

print(eval(testset))

model = model:float()  -- converting cuda model to cpu model
torch.save('models/modelv2.net',model)   -- saving model
