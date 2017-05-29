nn = require 'nn'
model = torch.load('models/initmodelv5.net')  -- loading the model
model
for i=6,40 do
  model:remove(6)
end
model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.ReLU())
model:add(nn.View(6*6*64))
model:add(nn.Dropout(.2))
model:add(nn.Linear(64*6*6,100))
model:add(nn.ReLU())
model:add(nn.Linear(100,17))
model:add(nn.LogSoftMax())
for i=1,5 do
  model.modules[i].train = false
end
torch.save('models/initmodelv5.net',model)
