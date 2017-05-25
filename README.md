# product_classification

| Model   | Description | Image Size | Accuracy | Training set | Iterations | Hyperparameters         | 
| --------|:-----------:| ----------:| -------- |:------------:| ----------:| -----------------------:|
| modelv1 | dropout     | 100        | 100      | 80%          | 500        | `lr=.0001`              | 
| modelv2 | dropout     | 100        | 100      | 80%          | 1000       | `lr=.0001` `lrdecy=0.09` | 
| modelv3 | vggnet16    | 224        | ---      | 80%          | 1000       | `lr=.0001` `lrdecy=0.09` |
| modelv4 | overfeat    | 231        | ---      | 80%          | 1000       | `lr=.0001` `lrdecy=0.09` |