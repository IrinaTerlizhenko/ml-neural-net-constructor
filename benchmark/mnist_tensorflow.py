import sys
import time

import numpy as np
import tensorflow as tf

# const
NUM_CLASSES = 10

# Data
eye = np.eye(NUM_CLASSES)


def build_labels(labels):
    return [eye[val] for val in labels]


def resize(val):
    tmp = np.zeros(NUM_CLASSES)
    tmp[val] = 1
    return tmp


def build_labels_v2(labels):
    return [resize(val) for val in labels]


# mnist
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# training
train_images = mnist.train.images
train_labels = mnist.train.labels

# validation
validation_images = mnist.validation.images
validation_labels = mnist.validation.labels

# test
test_images = mnist.test.images
test_labels = mnist.test.labels

# labels with size [10]
train_labels = build_labels(train_labels)
validation_labels = build_labels(validation_labels)
test_labels = build_labels(test_labels)

# const
num_features = len(train_images[0])
num_trainings = len(train_images)
num_validations = len(validation_images)
num_tests = len(test_images)

# data
with tf.name_scope('data'):
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, num_features))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, NUM_CLASSES))
    learn_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)

# params
with tf.name_scope('params'):
    W = tf.get_variable(
        initializer=tf.initializers.zeros(),
        name='W',
        shape=(num_features, NUM_CLASSES)
    )
    b = tf.get_variable(
        initializer=tf.initializers.zeros(),
        name='b',
        shape=NUM_CLASSES
    )

# loss function
with tf.name_scope('loss_function'):
    with tf.name_scope('soft_max'):
        soft_max = tf.nn.softmax(tf.matmul(x, W) + b)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(soft_max), reduction_indices=1))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(soft_max, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# optimization
with tf.name_scope('optimization'):
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()

# summary
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()


# run one round with specific parameters
def run_round(session, batch_size=100, learning_rate=0.01, num_epochs=20):
    session.run(init)
    global_step = 0
    for epoch in range(num_epochs):
        avg_cost = 0.
        total_batch = int(num_trainings / batch_size)

        for i in range(total_batch):
            batch_xs = train_images[i * batch_size:(i + 1) * batch_size]
            batch_ys = train_labels[i * batch_size:(i + 1) * batch_size]

            _, _, c = sess.run([merged_summary, optimizer, cross_entropy],
                               feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          learn_rate: learning_rate})
            global_step += 1

            avg_cost += c / total_batch

        print("Epoch:", '%04d' % (epoch + 1), ", cost:", "{:.9f}".format(avg_cost), file=sys.stderr)  # todo remove
    print("Optimization Finished!", file=sys.stderr)
    validation_accuracy = accuracy.eval({x: validation_images, y: validation_labels})
    w_const = W.eval()
    b_const = b.eval()
    print(f"Validation_Accuracy_{batch_size}_{learning_rate}_{num_epochs}: {validation_accuracy}", file=sys.stderr)

    solution = {"validation_accuracy": validation_accuracy, "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs, "W": w_const, "b": b_const}

    return solution


start = time.time()

with tf.Session() as sess:
    batch = 30
    rate = 0.05
    epochs = 30
    sol = run_round(sess, batch, rate, epochs)

end = time.time()
print(end - start)

with tf.Session() as sess:
    sess.run(W.assign(sol["W"]))
    sess.run(b.assign(sol["b"]))

    test_accuracy = accuracy.eval({x: test_images, y: test_labels})

print(
    f'Best Accuracy on Validation = {sol["validation_accuracy"]}: batch_size={sol["batch_size"]},'
    f' learn_rate={sol["learning_rate"]}, num_epochs={sol["num_epochs"]}')

print(
    f'Test Accuracy on Best algorithm = {test_accuracy}')

# Run time 40 minutes
# Best Accuracy on Validation = 0.9333999752998352: batch_size=30, learn_rate=0.05, num_epochs=30
# Test Accuracy on Best algorithm = 0.9277999997138977
