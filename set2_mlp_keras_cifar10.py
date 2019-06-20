# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of Sparse Evolutionary Training (SET) of Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.
# This implementation can be used to test SET in varying conditions, using the Keras framework versatility, e.g. various optimizers, activation layers, tensorflow
# Also it can be easily adapted for Convolutional Neural Networks or other models which have dense layers
# However, due the fact that the weights are stored in the standard Keras format (dense matrices), this implementation can not scale properly.
# If you would like to build and SET-MLP with over 100000 neurons, please use the pure Python implementation from the folder "SET-MLP-Sparse-Python-Data-Structures"

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
from sRelu import *
from keras.datasets import cifar10
from keras.utils import np_utils
import argparse
from random import randint
import copy
import datetime


class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


class Individual:

    def __init__(self, wm1, wm2, wm3,
                 mut_rate_wm1, mut_rate_wm2, mut_rate_wm3,
                 w1 = None, w2 = None, w3 = None, w4 = None
                 ):
        self.wm1 = wm1
        self.wm2 = wm2
        self.wm3 = wm3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.mut_rate_wm1 = mut_rate_wm1
        self.mut_rate_wm2 = mut_rate_wm2
        self.mut_rate_wm3 = mut_rate_wm3
        self.model = None
        self.fitness = None

    def mutate(self):
        self.wm1 = self.mutation(self.wm1, self.mut_rate_wm1)
        self.wm2 = self.mutation(self.wm2, self.mut_rate_wm2)
        self.wm3 = self.mutation(self.wm3, self.mut_rate_wm3)

    def mutation(self, wm, mut_rate):
        try_mut = np.random.rand(len(wm), len(wm[0]))
        flip = try_mut < mut_rate
        wm_flipped = np.absolute((wm-1)*-1)
        wm[flip] = wm_flipped[flip]

        return wm


class SET2_MLP_CIFAR10:

    def __init__(self, config):
        self.config = config
        # set model parameters
        self.epsilon = 20 # control the sparsity level as discussed in the paper
        self.zeta = 0.3 # the fraction of the weights removed
        self.batch_size = 100 # batch size
        self.maxepoches = 1000 # number of epochs
        self.learning_rate = 0.01 # SGD learning rate
        self.num_classes = 10 # number of classes
        self.momentum = 0.9 # SGD momentum
        self.pop_size = 10 # number of individuals in the evolving population
        self.epochs_per_generation = int(config.epochs_per_generation) # number of epochs to train solutions of a generation (naturally, number of generations is maxepoches/epochs_per_generation

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1, wm1, self.mut_rate_wm1] = self.createWeightsMask(self.epsilon, 32 * 32 * 3, 4000)
        [self.noPar2, wm2, self.mut_rate_wm2] = self.createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3, wm3, self.mut_rate_wm3] = self.createWeightsMask(self.epsilon, 1000, 4000)

        # initialize masks replicating the initial sparse masks
        self.population = [Individual(wm1.copy(), wm2.copy(), wm3.copy(),
                                      self.mut_rate_wm1, self.mut_rate_wm2, self.mut_rate_wm3
                                      ) for individual in range(0, self.pop_size)]

        # differentiate them by a mutation operation
        for individual in range(0, self.pop_size):
            self.population[individual].mutate()

        # create a SET-MLP model
        self.create_model()

        # train the SET-MLP model
        self.train()


    def create_model(self):

        for individual in range(0, self.pop_size):

            # create a SET-MLP model for CIFAR10 with 3 hidden layers
            self.population[individual].model = Sequential()
            self.population[individual].model.add(Flatten(input_shape=(32, 32, 3)))
            self.population[individual].model.add(Dense(4000,
                                                        name="sparse_1",  kernel_constraint=MaskWeights(self.population[individual].wm1),
                                                        weights=self.population[individual].w1))
            self.population[individual].model.add(SReLU(name="srelu1"))
            self.population[individual].model.add(Dropout(0.3))
            self.population[individual].model.add(Dense(1000,
                                                        name="sparse_2", kernel_constraint=MaskWeights(self.population[individual].wm2),
                                                        weights=self.population[individual].w2))
            self.population[individual].model.add(SReLU(name="srelu2"))
            self.population[individual].model.add(Dropout(0.3))
            self.population[individual].model.add(Dense(4000,
                                                        name="sparse_3",  kernel_constraint=MaskWeights(self.population[individual].wm3),
                                                        weights=self.population[individual].w3))
            self.population[individual].model.add(SReLU(name="srelu3"))
            self.population[individual].model.add(Dropout(0.3))
            self.population[individual].model.add(Dense(self.num_classes, name="dense_4",
                                                        weights=self.population[individual].w4))

            #please note that there is no need for a sparse output layer as the number of classes is much smaller than the number of input hidden neurons
            self.population[individual].model.add(Activation('softmax'))

    def find_first_pos(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def find_last_pos(self, array, value):
        idx = (np.abs(array - value))[::-1].argmin()
        return array.shape[0] - idx

    def createWeightsMask(self, epsilon, noRows, noCols):
        # generate an Erdos Renyi sparse weights mask
        mask_weights = np.random.rand(noRows, noCols)
        p = (epsilon * (noRows + noCols)) / (noRows * noCols)
        prob = 1 - p  # normal tp have 8x connections
        mask_weights[mask_weights < prob] = 0
        mask_weights[mask_weights >= prob] = 1
        noParameters = np.sum(mask_weights)

        return [noParameters, mask_weights, p]

    def update_population(self):

        for individual in range(0, self.pop_size):
            self.population[individual].w1 = self.population[individual].model.get_layer("sparse_1").get_weights()
            self.population[individual].w2 = self.population[individual].model.get_layer("sparse_2").get_weights()
            self.population[individual].w3 = self.population[individual].model.get_layer("sparse_3").get_weights()
            self.population[individual].w4 = self.population[individual].model.get_layer("dense_4").get_weights()

    def tournament_selection(self, k=2):
        """
        Perform tournament selection and return best individual
        :param k: amount of individuals to participate in tournament
        """
        best_individual = None
        for _ in range(k):
            individual_index = randint(0, len(self.population) - 1)
            if (best_individual is None) \
                    or (self.population[individual_index].fitness > self.population[best_individual].fitness):
                best_individual = individual_index
        return best_individual

    def cross(self, wm_p1, wm_p2, w_p1, w_p2):

        nlines = len(w_p1)
        ncols = len(w_p1[0])
        points = np.random.uniform(0, 1, nlines)
        points = np.repeat(points, ncols)
        points = points.reshape(nlines, ncols) * ncols
        indices_aux = np.tile(np.arange(ncols), nlines).reshape(nlines, ncols)

        w = np.where(indices_aux < points, w_p1, w_p2)
        if wm_p1 is not None:
            wm = np.where(indices_aux < points, wm_p1, wm_p2)

        if wm_p1 is not None:
            return wm, w
        else:
            return w

    def evolve(self, individuals_accuracies):

        offspring = []

        for individual in range(0, self.pop_size):
            self.population[individual].fitness = individuals_accuracies[individual]

        for individual in range(0, self.pop_size):
            parent1 = self.tournament_selection()
            parent2 = parent1
            while parent2 == parent1:
                parent2 = self.tournament_selection()
            #print('new individual '+str(individual)+' has parent1 '+str(parent1)+' and parent2 '+str(parent2))

            w1 = copy.copy(self.population[parent1].w1)
            w2 = copy.copy(self.population[parent1].w2)
            w3 = copy.copy(self.population[parent1].w3)
            w4 = copy.copy(self.population[parent1].w4)

            wm1, w1[0] = self.cross(self.population[parent1].wm1,
                                    self.population[parent2].wm1,
                                    self.population[parent1].w1[0],
                                    self.population[parent2].w1[0])

            wm2, w2[0] = self.cross(self.population[parent1].wm2,
                                    self.population[parent2].wm2,
                                    self.population[parent1].w2[0],
                                    self.population[parent2].w2[0]
                                    )

            wm3, w3[0] = self.cross(self.population[parent1].wm3,
                                    self.population[parent2].wm3,
                                    self.population[parent1].w3[0],
                                    self.population[parent2].w3[0])


            w4[0] = self.cross(None, None,
                               self.population[parent1].w4[0],
                               self.population[parent2].w4[0])

            new_individual = Individual(wm1, wm2, wm3,
                                        self.mut_rate_wm1, self.mut_rate_wm2, self.mut_rate_wm3,
                                        w1, w2, w3, w4
                                        )
            new_individual.mutate()

            offspring.append(new_individual)

        self.population = offspring

    def train(self):

        individual = 0

        # read CIFAR10 data
        [x_train,x_test,y_train,y_test]=self.read_data()

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # shows only one, as the topology is always equal
        self.population[0].model.summary()

        # training process in a for loop

        self.best_accuracies = []
        self.all_accuracies = []
        for epoch in range(0, self.maxepoches):

            individuals_accuracies = []
            for individual in range(0, self.pop_size):

                sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
                self.population[individual].model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

                historytemp = self.population[individual].model.fit_generator(datagen.flow(x_train, y_train,
                                                                                           batch_size=self.batch_size),
                                                                              steps_per_epoch=x_train.shape[0]//self.batch_size,
                                                                              epochs=epoch,
                                                                              validation_data=(x_test, y_test),
                                                                              initial_epoch=epoch-1)
                individuals_accuracies.append(historytemp.history['val_acc'][0])
                self.all_accuracies.append(historytemp.history['val_acc'][0])


            self.best_accuracies.append(max(individuals_accuracies))

            print(' \n >>> Best acc '+ str(round(self.best_accuracies[-1], 4)) + ' \n')
            file = open("results_"+config.exp_name+"/set2_mlp_srelu_sgd_cifar10_acc.txt",'a')
            file.write(str(round(self.best_accuracies[-1], 4))+ '\n')
            file.close()

            self.update_population()

            # evolves models after a number of epochs
            if (epoch+1) % self.epochs_per_generation == 0:
                print(' \n >>> Evolution step <<< \n')
                self.evolve(individuals_accuracies)

            #ugly hack to avoid tensorflow memory increase for multiple fit_generator calls. Theano shall work more nicely this but it is outdated in general
            K.clear_session()

            self.create_model()

        for individual in range(0, self.pop_size):
            print('Individual ' + str(individual))
            print(' n. params of wm1 ' + str(np.sum(self.population[individual].wm1)))
            print(' n. params of wm2 ' + str(np.sum(self.population[individual].wm2)))
            print(' n. params of wm3 ' + str(np.sum(self.population[individual].wm3)))


    def read_data(self):

        #read CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        sample_size = int(len(x_train)* float(self.config.sample_prop) )
        x_train = x_train[0:sample_size]
        y_train = y_train[0:sample_size]
        x_test = x_test[0:sample_size]
        y_test = y_test[0:sample_size]

        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #normalize data
        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTtrainStd
        x_test = (x_test - xTrainMean) / xTtrainStd

        return [x_train, x_test, y_train, y_test]

if __name__ == '__main__':


    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M"))

    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_prop',
                        default=0.02)

    parser.add_argument('--exp_name',
                        default='new')

    parser.add_argument('--epochs_per_generation',
                        default=20)


    config = parser.parse_args()

    # create and run a SET-MLP model on CIFAR10
    model=SET2_MLP_CIFAR10(config)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M"))

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results_" + config.exp_name + "/set2_mlp_srelu_sgd_cifar10_acc_all.txt", np.asarray(model.all_accuracies))




