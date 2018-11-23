#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    if len(x.shape) > 0:
        s = np.divide(1,np.add(1,np.exp(np.multiply(-1,x),dtype=np.float64)),dtype=np.float64)
    else:
        s =  1 / (1 + np.exp(-1 * x))
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    if len(s.shape) > 0:
        temps = np.subtract(1, s)
        ds = np.multiply(temps, s, dtype=np.float64)
    else:
        ds = s * (1 - s)
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    x = np.array(1.0)
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array(0.7310585786300049)
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array(0.19661193324148185)
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"

    x = np.array([-0.07530336,0.16462545,0.36824303])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([0.48118302,0.54106367,0.59103435])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([0.2496459,0.24831377,0.24171275])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"

    ### END YOUR CODE


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
