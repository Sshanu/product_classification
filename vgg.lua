optim = require 'optim'
require 'cunn';

fullset = torch.load('data/size_224.dat') -- Shuffling fullset
size = fullset.size
shuffle = torch.randperm(size)
shuffleset = fullset
for i=1,size do
    shuffleset.data[i] = fullset.data[shuffle[i]]
    shuffleset.label[i] = fullset.label[shuffle[i]]
    end
fullset = shuffleset

-- Trainset 50% of fullset
trainset = {
    data = fullset.data[{{1,size*.5}}]:float(),
    label = fullset.label[{{1,size*.5}}],
    size = size*.5
}

-- Testset 20% of fullset
testset = {
    data = fullset.data[{{size*.5+1,size}}]:float(),
    label = fullset.label[{{size*.5+1,size}}],
    size = size-size*.5
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

model = torch.load('initmodelv3.net')  -- loading the model
model = model:cuda()  -- model for gpu
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainset.data = trainset.data:cuda()    -- Trainset for cuda
trainset.label = trainset.label:cuda()

trainer = nn.StochasticGradient(model, criterion) --Training hyperparameters
trainer.learningRate = 0.0001
trainer.maxIteration = 500
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
torch.save('modelv3.net',model)   -- saving model
