from __future__ import print_function

import os
import sys
import timeit
import gzip
import pickle

import numpy
import cPickle

import theano
import theano.tensor as T


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
   
    max_shape = 5000
    v_shape = 2000
    test_set = (test_set[0][:max_shape],test_set[1][:max_shape])
    valid_set = (valid_set[0][:v_shape],valid_set[1][:v_shape])
    train_set = (train_set[0][:v_shape],train_set[1][:v_shape])
    

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


class FcLayer(object):
    def __init__(self, rng, fp_input, bp_input, n_up, n_down, 
                 W = None, 
                 fp_activation=T.nnet.sigmoid,
                 bp_activation=T.nnet.sigmoid,
                 ):
        self.fp_input = fp_input
        self.bp_input = bp_input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low= -1,
                    high= 1,
                    size=(n_up, n_down)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)


        self.w_fp = W
        self.w_bp = W.T

        lin_fp_output = T.dot(fp_input, self.w_fp) 
        lin_bp_output = T.dot(bp_input, self.w_bp) 


        self.fp_output = (
            lin_fp_output if fp_activation is None
            else fp_activation(lin_fp_output)
        )

        self.bp_output = (
            lin_bp_output if bp_activation is None
            else bp_activation(lin_bp_output)
        )


class MLP(object):

    def __init__(self, rng, fp_input, bp_input):

        self.layer0 = FcLayer(
            rng=rng,
            fp_input=fp_input,
            bp_input=bp_input,
            n_up=784,
            n_down=20,
        )

        self.output = self.layer0.fp_output


def main(dataset='../data/mnist.pkl.gz', batch_size=20, n_epochs=10):


    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()  
    x = T.matrix('x') 
    y = T.ivector('y')
    bp_x = T.matrix('bp_x') 

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        fp_input=x,
        bp_input=bp_x
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.output,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            res = test_model(minibatch_index)
            print(res)

    
if __name__ == '__main__':
    main()
