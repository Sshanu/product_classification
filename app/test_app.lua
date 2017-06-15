torch = require "torch"
nn = require "nn"
image = require "image"
id = arg[1]
image_name = arg[2]
dirs = {}

for file in paths.files(id..'/images/') do
    if file ~= "." then
        if file ~= ".." then
            table.insert(dirs, file)
        end
    end
end
table.sort(dirs, function (a,b) return a < b end)
model_name = id..'/models/model_'..id..'.net'
print(dirs)
model = torch.load(model_name)
img = image.load(id..'/test/'..image_name, 3, 'byte')
img = image.scale(img, 48, 48):double()
prediction = model:forward(img)
confidences, indices = torch.sort(prediction, true)
for i=1,3 do
  print(dirs[indices[i]], "label "..indices[i],"score "..confidences[i] )
end
