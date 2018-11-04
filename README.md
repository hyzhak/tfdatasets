# tfdatasets
middleware in pipeline between dataset and tensorflow classifier


## Usage:

### CIFAR-10

```python
import tensorflow as tf
import tfdatasets

ds = tfdatasets.get_dataset('cifar10', path = '/var/datasets')

# create simple DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=ds.get_all_feature_columns(),
    # Two hidden layers of 10 nodes each.
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    dropout=0.1,
    n_classes=10,
    model_dir='/var/models/dnn-256-32')

# train it
classifier.train(input_fn=ds.train(), steps=1000)
# validate
classifier.evaluate(input_fn=ds.validation())
# test on test sub-data set 
classifier.evaluate(input_fn=ds.test())

```

### Iris

```python
import tensorflow as tf
import tfdatasets

ds = tfdatasets.get_dataset('iris', path = '/var/datasets')

# TODO:
# Feature columns describe how to use the input.


classifier = tf.estimator.DNNClassifier(
    feature_columns=ds.get_all_feature_columns(),
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=ds.num_of_classes())

# train it
classifier.train(input_fn=ds.train(), steps=1000)
# validate
classifier.evaluate(input_fn=ds.validation())
# test on test sub-data set
classifier.evaluate(input_fn=ds.test())

```

## Analogs:
- https://github.com/stefanwebb/tensorflow-datasets
