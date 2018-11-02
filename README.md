# tfdatasets
middleware in pipeline between dataset and tensorflow classifier


## Usage:

```python
import tensorflow as tf
import tfdatasets

ds = tfdatasets.get_dataset('cifar10', path = '/var/datasets')

# create simple DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column('image', shape=[3, 32, 32])],
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

## Analogs:
- https://github.com/stefanwebb/tensorflow-datasets
