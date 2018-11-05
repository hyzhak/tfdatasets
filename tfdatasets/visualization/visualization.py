import tensorflow as tf
import matplotlib.pyplot as plt


def show_samples(ds):
    iterator = ds.batch(32).make_one_shot_iterator()
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
