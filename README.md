# product_classification

## Training 

```
th train.lua model_name dataset_name maxiterations
```
example:

```
th train.lua modelv7.net size_48.dat 100
```

## Testing

```
th test.lua model_name image_name
```
It will return top three predicted labels with their scores

example:

```
th test.lua modelv7.net image.jpg
```
## Model
Neural Network Architecture for input image size :48*48
```
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
```


## Experiments

| Model   | Description      | Image Size | Accuracy | Training set | Iterations | Hyperparameters          |
| --------|:----------------:| ----------:| -------- |:------------:| ----------:| -----------------------: |
| modelv1 | dropout          | 100        | 100      | 80%          | 500        | `lr=.0001`               |
| modelv2 | dropout          | 100        | 100      | 80%          | 1000       | `lr=.0001` `lrdecy=0.09` |
| modelv3 | vggnet16         | 224        | <20      | 50%          | 500        | `lr=.0001` `lrdecy=0.09` |
| modelv4 | overfeat         | 231        | <20      | 50%          | 500        | `lr=.0001` `lrdecy=0.09` |
| modelv5 | vgg16 5layer     |  48        | 100      | 50%          | 500        | `lr=.001`                |
| modelv6 | vgg16 5layer(f)  |  48        | 100      | 50%          | 500        | `lr=.001`                |
| modelv7 | dropout          |  48        | 100       | 100%(basic)  | 200        | `lr=.001`                |

modelv7 trained by sgd 100% accuracy in 200 iterations
modelv7 trained by adam 100% accuracy in 20 iterations
f = train fasle for given layers
