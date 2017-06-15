fullset = torch.load('data/basic_size_48.dat') -- Shuffling fullset
size = fullset.size
shuffle = torch.randperm(size)
shuffleset = fullset
for i=1,size do
    shuffleset.data[i] = fullset.data[shuffle[i]]
    shuffleset.label[i] = fullset.label[shuffle[i]]
    end

-- Trainset 80% of fullset
trainset = {
    data = fullset.data[{{1,size}}]:double(),
    label = fullset.label[{{1,size}}],
    size = size
}
img = image.load('data/images/a.jpg',3, 'byte')
img = image.scale(img, 48, 48):double()
data_ = torch.Tensor(1, 3, 48, 48)
data_[1] = img
label_ = torch.Tensor(1)
label_[1] = 11
trainset_ = {
  data = data_,
  label = label_,
  size = 1
}
data_test
testset = {
  data
}
optim = require "optim"
nn = require "nn"

model = torch.load('models/initmodelv7.net')
criterion = nn.ClassNLLCriterion()


adam_params = {
   learningRate = 0.0001,
}

x, dl_dx = model:getParameters()

step = function(dataset, batch_size)
    local current_loss = 0
    local count = 0
    batch_size = batch_size
    for t = 1, dataset.size, batch_size do
        local size = math.min(t + batch_size, dataset.size + 1) - t
        inputs = torch.Tensor(size, 3, 48, 48)
        targets = torch.Tensor(size)
        for i = 1,size do
            local input = dataset.data[i+t-1]
            local target = dataset.label[i+t-1]
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target
        end

        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end

        _, fs = optim.adam(feval, x, adam_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end
eval = function(dataset, batch_size)
    count = 0
    batch_size = batch_size

    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size, dataset.size + 1) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size * 100
end


max_iters = 50

do
    local Plot = require 'itorch.Plot'
    local last_accuracy = 0
    local decreasing = 0
    local loss={}
    local x={}
    local accuracy={}
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        x[i]=i
        loss[i]=step(trainset_, 1)
        print(string.format('Epoch: %d Current loss: %4f', i, loss[i]))
        accuracy[i] =eval(trainset_, 1)
        print(string.format('Accuracy on the validation set: %4f', accuracy[i]))
        -- if accuracy[i] == 100 then
        --   print("100% Accuracy reached")
        --   break
        -- end
    end

end
torch.save('models/modelv8.net', model)
