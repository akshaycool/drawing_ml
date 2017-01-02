import pickle
import fnmatch
import os
import cPickle
import numpy as np

def get_unpickled_file(file):
    fo = open(file,'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_training_set():
    data_files = []
    for file in os.listdir(root_dir):
        if fnmatch.fnmatch(file,'data_batch*'):
            data_files.append(file)

    X = []
    Y = []
    for file in data_files:
        print "file",file
        data_dict = (get_unpickled_file(root_dir+"/"+str(file)))
        print "data_dict",len(data_dict['data'])
        data = data_dict['data']
        labels = data_dict['labels']
        data_list = data.tolist()
        X.append(data_list)
        Y = Y + labels

    X = np.vstack(X)
    Y = np.asarray(Y)
    print len(X),len(Y)
    return X,Y

if __name__ == '__main__':
    root_dir = 'data/cifar-10-batches-py'
    X ,Y = get_training_set()
    assert len(X) == 50000
    assert len(Y) == 50000

