torch = require "torch"
nn = require "nn"

model_name = arg[1]
dataset_name = arg[2]
max_iteration = tonumber(arg[3])

-- loading training set
trainset = torch.load('data/'..dataset_name)

-- Shuffling the training set
shuffle = torch.randperm(trainset.size)
shuffleset = trainset
print(trainset)
size = trainset.size
for i=1, trainset.size do
  shuffleset.data[i] = trainset.data[shuffle[i]]
  shuffleset.label[i] = trainset.label[shuffle[i]]
end


-- Converting trainset to meatable
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end


-- loading model
model = torch.load('models/'..model_name)
criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(model, criterion) --Training hyperparameters
trainer.learningRate = 0.001
trainer.maxIteration = max_iteration
trainer:train(trainset)

torch.save('models/'..model_name, model)

eval = function(dataset)      -- evalutation of testset
    correct = 0
    for i=1,size do
        local target = dataset.label[i]
        local prediction = model:forward(dataset.data[i])
        local confidences, indices = torch.sort(prediction, true)
        if target == indices[1] then
            correct = correct + 1
        end
    end
    return correct/size*100
end

print(eval(trainset))
