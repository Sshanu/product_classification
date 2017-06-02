# product_classification

| Model   | Description      | Image Size | Accuracy | Training set | Iterations | Hyperparameters          |
| --------|:----------------:| ----------:| -------- |:------------:| ----------:| -----------------------: |
| modelv1 | dropout          | 100        | 100      | 80%          | 500        | `lr=.0001`               |
| modelv2 | dropout          | 100        | 100      | 80%          | 1000       | `lr=.0001` `lrdecy=0.09` |
| modelv3 | vggnet16         | 224        | <20      | 50%          | 500        | `lr=.0001` `lrdecy=0.09` |
| modelv4 | overfeat         | 231        | <20      | 50%          | 500        | `lr=.0001` `lrdecy=0.09` |
| modelv5 | vgg16 5layer     |  48        | 100      | 50%          | 500        | `lr=.001`                |
| modelv6 | vgg16 5layer(f)  |  48        | 100      | 50%          | 500        | `lr=.001`                |
| modelv7 | dropout          |  48        | 75       | 100%(basic)  | 100        | `lr=.001`                |
f = train fasle for given layers