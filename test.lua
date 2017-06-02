image_name = arg[2]
model_name = arg[1]

model = torch.load('models/'..model_name)
image = image.load(image_name, 3, 'byte')
image = image.scale(image, 48, 48)
prediction = model:forward(image)
confidences, indices = torch.sort(prediction, true)
for i in indices do
  print(confidences[i],i)
end
