import tensorflow as tf
from PIL import Image
import numpy
from image_flow import read_images
from tensorflow.examples.tutorials.mnist.input_data import dense_to_one_hot, DataSet

def shuffle_in_unison_scary(a, b):
    '''http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison'''
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)


class CustomDataSet(DataSet):

    def __init__(self, images, labels, fake_data=False, one_hot=False):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                    labels.shape))
        self._num_examples = images.shape[0]
        # # Convert from [0, 255] -> [0.0, 1.0].
        # images = images.astype(numpy.float32)
        # images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0


def main():
    day_images = read_images('/Users/jason/Documents/day_night_detection/day_files/')
    day_labels = dense_to_one_hot(numpy.zeros((numpy.asarray(day_images).shape[0],), dtype=numpy.uint8), num_classes=2)
    night_images = read_images('/Users/jason/Documents/day_night_detection/night_files/')
    night_labels = dense_to_one_hot(numpy.ones((numpy.asarray(night_images).shape[0],), dtype=numpy.uint8), num_classes=2)


    all_images = numpy.append(day_images, night_images, axis=0)
    all_labels = numpy.append(day_labels,night_labels, axis=0)

    shuffle_in_unison_scary(all_images, all_labels)

    train, test = numpy.array_split(all_images, 2)
    train_l, test_l = numpy.array_split(all_labels, 2)


    train_data = CustomDataSet(train, train_l)
    test_data = CustomDataSet(test, test_l)

    print("Datasets Created, lengths per set: {}".format(len(test)))

    x = tf.placeholder(tf.float32, [None, 6220800])
    W = tf.Variable(tf.zeros([6220800, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(500):
        print(i)
        batch_xs, batch_ys = train_data.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels}))



    # # Create the model
    # x = tf.placeholder("float", [None, 6220800], name="x-input")
    # W = tf.Variable(tf.zeros([6220800,2]), name="weights")
    # b = tf.Variable(tf.zeros([2], name="bias"))
    #
    # # use a name scope to organize nodes in the graph visualizer
    # with tf.name_scope("Wx_b") as scope:
    #   y = tf.nn.softmax(tf.matmul(x,W) + b)
    #
    # # Add summary ops to collect data
    # w_hist = tf.histogram_summary("weights", W)
    # b_hist = tf.histogram_summary("biases", b)
    # y_hist = tf.histogram_summary("y", y)
    #
    # # Define loss and optimizer
    # y_ = tf.placeholder("float", [None,2], name="y-input")
    # # More name scopes will clean up the graph representation
    # with tf.name_scope("xent") as scope:
    #   cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    #   ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
    #
    # with tf.name_scope("train") as scope:
    #   train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # with tf.name_scope("test") as scope:
    #   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #   accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    #
    # # Merge all the summaries and write them out to /tmp/mnist_logs
    # merged = tf.merge_all_summaries()
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # writer = tf.train.SummaryWriter("/tmp/tf_logs", sess.graph_def)
    #
    # # Train the model, and feed in test data and record summaries every 10 steps
    #
    # for i in range(1000):
    #   if i % 10 == 0:  # Record summary data, and the accuracy
    #     feed = {x: test_data.images, y_: test_data.labels}
    #     result = sess.run([merged, accuracy], feed_dict=feed)
    #     summary_str = result[0]
    #     acc = result[1]
    #     writer.add_summary(summary_str, i)
    #     print("Accuracy at step %s: %s" % (i, acc))
    #   else:
    #     batch_xs, batch_ys = train_data.next_batch(50)
    #     feed = {x: batch_xs, y_: batch_ys}
    #     sess.run(train_step, feed_dict=feed)
    #
    # print(accuracy.eval({x: test_data.images, y_: test_data.labels}))




main()


