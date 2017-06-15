image = require 'image'
nn = require "nn"
optim = require "optim"
torch = require "torch"
img_size = 48
id = arg[1]
dirs = {}
files = {}
labels = {}

ext ='jpg'

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

for file in paths.files(id..'/images/') do
    if file ~= "." then
        if file ~= ".." then
            table.insert(dirs, file)
        end
    end
end
table.sort(dirs, function (a,b) return a < b end)
no_dirs = tablelength(dirs)
print(no_dirs)

for i, dir in pairs(dirs) do
    for file in paths.files(id..'/images/'..dir) do
   -- We only load files that match the extension
       if file:find(ext .. '$') then
          -- and insert the ones we care about in our table
          table.insert(files, id..'/images/'..dir..'/'..file)
          table.insert(labels,i)
        end
    end
end

size = tonumber(tablelength(files))
ibt=torch.ByteTensor(size,3,img_size,img_size)
il=torch.ByteTensor(size)

for i,file in pairs(files) do
    img = image.load(file,3,'byte')
    img = image.scale(img,img_size,img_size)
    ibt[i] = img
    il[i] = labels[i]
end


trainset = {data=ibt:double(), label=il, size = size}

model = torch.load('initial_model.net')
model:add(nn.Linear(1296, no_dirs))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

adam_params = {
   learningRate = 0.0001}

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


num_iters = (no_dirs/10) * 20
max_iters = num_iters * 10
print(num_iters)
do
    for i = 1,max_iters do
        local loss = step(trainset, 1)
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(trainset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if (i%num_iters==0)then
          torch.save(id..'/models/model_'..id..'.net', model)
          if accuracy >= 99 then
            print("100% Accuracy reached")
            break
          end
        end
    end
end
torch.save(id..'/models/model_'..id..'.net', model)
