"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data. For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``. In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.

Note that the code requires the file ``../data/mnist.pkl``. If it's
not already in that directory then you should unzip the file
``../data/mnist.pkl.gz``.
"""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


def load_data():
    """Return the MNIST data as a tuple containing the training data,
the validation data, and the test data.

The ``training_data`` is returned as a tuple with two entries.
The first entry contains the actual training images. This is a
numpy ndarray with 50,000 entries. Each entry is, in turn, a
numpy ndarray with 784 values, representing the 28 * 28 = 784
pixels in a single MNIST image.

The second entry in the ``training_data`` tuple is a numpy ndarray
containing 50,000 entries. Those entries are just the digit
values (0...9) for the corresponding images contained in the first
entry of the tuple.

The ``validation_data`` and ``test_data`` are similar, except
each contains only 10,000 images.

This is a nice data format, but for use in neural networks it's
helpful to modify the format of the ``training_data`` a little.
That's done in the wrapper function ``load_data_wrapper()``, see
below.
"""
    f = open('data/mnist.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
test_data)``. Based on ``load_data``, but the format is more
convenient for use in our implementation of neural networks.

In particular, ``training_data`` is a list containing 50,000
2-tuples ``(x, y)``. ``x`` is a 784-dimensional numpy.ndarray
containing the input image. ``y`` is a 10-dimensional
numpy.ndarray representing the unit vector corresponding to the
correct digit for ``x``.

``validation_data`` and ``test_data`` are lists containing 10,000
2-tuples ``(x, y)``. In each case, ``x`` is a 784-dimensional
numpy.ndarry containing the input image, and ``y`` is the
corresponding classification, i.e., the digit values (integers)
corresponding to ``x``.

Obviously, this means we're using slightly different formats for
the training data and the validation / test data. These formats
turn out to be the most convenient for use in our neural network
code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def load_training_data_with_label(label):
    '''
Return training data with a particular label
'''
    tr_d, _, _ = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    tr_d2 = [training_data[i] for i in xrange(len(training_data)) if not np.any(training_data[i][1]-vectorized_result(label))]   
        
    return tr_d2
    


def display_training_data(label, num_examples=10):
    '''
Show training data with a particular label as an image
'''
    td = load_training_data_with_label(label)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(td[0],(28,28)), cmap=cm.Greys_r)
    fig.show()
    
    for i in xrange(1,min([num_examples, len(td)])):
        im.set_data(np.reshape(td[i],(28,28)))
        fig.canvas.draw()
        plt.pause(0.1)


def display_pca(M, cov):
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(M,(28,28)), cmap=cm.Greys_r)
    plt.title('Original image')
    
    # Perform SVD: cov = U*S*V'
    # Note that U == V since cov matrix is symmetric
    U, s, _ = svd(cov, full_matrices=False)
    
    # Sort by descending order of singular values
    ind = np.argsort(s)[::-1]
    U = U[:,ind]
    s = s[ind]
    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(U[:,0],(28,28)), cmap=cm.Greys_r)
    plt.title('Image from PCs')


def compute_stats_from_examples(examples):
    num_examples = len(examples)
    
    avg_examples = examples[0]
    cov_examples = np.outer(examples[0], examples[0])
    for i in xrange(num_examples):
        avg_examples += examples[i]
        cov_examples += np.outer(examples[i], examples[i])

    return {'mean': avg_examples / num_examples,
             'cov': cov_examples / num_examples}

        
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
position and zeroes elsewhere. This is used to convert a digit
(0...9) into a corresponding desired output from the neural
network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e