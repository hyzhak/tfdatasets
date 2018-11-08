import pandas as pd
import tensorflow as tf


def show_samples(ds, limit=32):
    """
    Read data from Tensorflow dataset to Pandas dataframe

    :param ds:
    :param limit:
    :return:
    """
    batch_iterator = ds.batch(limit).make_one_shot_iterator()
    with tf.Session() as sess:
        batch = batch_iterator.get_next()
        examples = sess.run(batch)[0]

    return pd.DataFrame.from_dict(examples)
