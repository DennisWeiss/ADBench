import numpy as np
from data_generator import DataGenerator

data_generator = DataGenerator(dataset='CIFAR10_0')

data = data_generator.generator(la=0, at_least_one_labeled=False)

print(data['y_test'])
print(data['y_test'].shape)
print(np.sum(1 - data['y_test']))

print(data['y_train'])
print(data['y_train'].shape)
print(np.sum(1 - data['y_train']))