torch = require "torch"
nn = require "nn"
image = require "image"
image_name = arg[2]
model_name = arg[1]

model = torch.load('models/'..model_name)
img = image.load(image_name, 3, 'byte')
img = image.scale(img, 48, 48):double()
prediction = model:forward(img)
confidences, indices = torch.sort(prediction, true)
for i=1,3 do
  print("label "..indices[i],"score "..confidences[i] )
end
