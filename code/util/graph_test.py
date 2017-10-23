from util import graph
import numpy as np
import tensorflow as tf
import unittest

class TestAdder(unittest.TestCase):

    def test_adder(self):
        add_two = graph.AddTwo()
        self.assertEqual(3, add_two.Add(1, 2))
        self.assertEqual(7, add_two.Add(2, 5))

    def test_vector_adder(self):
        add_two = graph.AddTwo()
        np.testing.assert_array_equal(
                np.array([5, 7, 9]),
                add_two.Add(
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6])))

class TestLayer(tf.test.TestCase):

    def test_affine(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            z = graph.affine_layer(10, x)
            # Verify graph properties.
            self.assertAllEqual(10, z.get_shape()[-1])

            # Setup for running the graph.
            sess.run(tf.global_variables_initializer())

            # Verify that dimensions work with more than one row.
            sess.run(z, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            # Verify computation correct.
            x_val = np.array([[3., 2., 1.]])
            z_val = sess.run(z, feed_dict={x: x_val})
            self.assertEqual((1, 10), z_val.shape)
            self.assertAllClose([[
                -0.92742872,  2.20097542,  0.72329885,
                -0.8619436 ,  -0.09275365, 1.63259518,
                3.04762506,   0.80803722,  0.48845479,
                2.23918748]], z_val)


    def test_fully_connected_layers(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            out = graph.fully_connected_layers([10, 20, 100, 1], x)
            self.assertAllEqual(1, out.get_shape()[-1])

            sess.run(tf.global_variables_initializer())

            sess.run(out, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            x_val = np.array([[-3., 2., 1.], [5., 6., 87.]])
            out_val = sess.run(out, feed_dict={x: x_val})
            self.assertEqual((2, 1), out_val.shape)
            self.assertAllClose([[1.735874],[50.472717]], out_val)

    def test_no_fully_connected_layers(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=[None, 3])
            out = graph.fully_connected_layers([], x)
            self.assertAllEqual(3, out.get_shape()[-1])

            sess.run(tf.global_variables_initializer())

            sess.run(out, feed_dict={
                x: np.array([[1., 2., 3.], [4., 5., 6.]])})

            x_val = np.array([[3., 2., 1.], [5., 6., 87.]])
            out_val = sess.run(out, feed_dict={x: x_val})
            self.assertEqual((2, 3), out_val.shape)
            self.assertAllClose(x_val, out_val)


class TestNN(unittest.TestCase):

    def test_train_nn(self):
        X_train, y_train, X_test, y_test = generate_data(1000, 10)
        y_model = graph.train_nn(X_train, y_train, X_test,
                [], 50, 2, 0.001)

    def test_train_nn_with_fclayers(self):
        X_train, y_train, X_test, y_test = generate_data(1000, 10)
        y_model = graph.train_nn(X_train, y_train, X_test,
                [10, 22, 37], 50, 2, 0.001)


def generate_data(num_train, num_test):
    np.random.seed(1)
    num = num_train + num_test
    x0 = np.random.randn(num, 2) + 3.*np.array([1, 0])
    x1 = np.random.randn(num, 2) + 1.*np.array([-1, 0])
    X = np.vstack([x0, x1])
    y = np.concatenate([
        np.zeros(num), np.ones(num)])

    # Randomly shuffle the data
    shuf_idx = np.random.permutation(len(y))
    X = X[shuf_idx]
    y = y[shuf_idx]

    return X[:num_train], y[:num_train], X[num_train:], y[num_train:]


def generate_non_linear_data(num_train, num_test):
    np.random.seed(1)
    num = num_train + num_test
    x0 = np.random.randn(num, 2) + 4.*np.array([1, 0])
    x1 = np.random.randn(num, 2) + 4.*np.array([0, 1])
    x2 = np.random.randn(num, 2) + 4.*np.array([-1, 0])
    x3 = np.random.randn(num, 2) + 4.*np.array([0, -2])
    X = np.vstack([x0, x1, x2, x3])
    y = np.concatenate([
        np.zeros(num), np.ones(num),
        np.zeros(num), np.ones(num)])

    # Randomly shuffle the data
    shuf_idx = np.random.permutation(len(y))
    X = X[shuf_idx]
    y = y[shuf_idx]

    return X[:num_train], y[:num_train], X[num_train:], y[num_train:]


if __name__ == '__main__':
    unittest.main()
