import tensorflow as tf
import matplotlib.pyplot as plt


def make_iterator(ds, batch_size=32, shuffle_size=1000):
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_size)
    return ds.make_one_shot_iterator()


def show_samples(ds):
    iterator = make_iterator(ds)
    with tf.Session() as sess:
        next_element = iterator.get_next()
        sess.run(tf.global_variables_initializer())
        examples = sess.run(next_element)

        grid_width = 16
        rows = 9
        items_in_a_row = 8
        plt.figure(figsize=(grid_width, grid_width / items_in_a_row * rows))
        for idx, (image, label) in enumerate(zip(examples[0]['image'], examples[1])):
            plt.subplot(rows, items_in_a_row, idx + 1)
            plt.title(label)
            plt.xticks([])
            plt.yticks([])
            # for some reasons matplotlib convert color data is inverted
            plt.imshow(image / 256., interpolation='nearest')
        plt.show()
