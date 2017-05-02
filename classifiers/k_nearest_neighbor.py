import numpy as np
from collections import Counter

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """


    num_test = X.shape[0]#500
    num_train = self.X_train.shape[0]#5000
    dists = np.zeros((num_test, num_train))

    # for i in xrange(num_test):
    #   # find the nearest training image to the i'th test image
    #   # using the L1 distance (sum of absolute value differences)
    #   print self.X_train ,X[i], X[i].shape
    #   print "indexed",X[i,:]
    #   distances = np.sum(np.abs(self.X_train - X[i,:]), axis = 1)
    #   break
    # print distances



    
    for i in xrange(num_test):
      dist_sum = 0
      for j in xrange(num_train):
        ####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        ####################################################################
        #print self.X_train[j].shape,self.X_train[j]
        #print X[i].shape,X[i]
        #break
        #print i,j
        x1 = np.square(X[i])
        x2 = np.square(self.X_train[i])
        test_i  = X[i, :]
        train_j = self.X_train[j, :]
        dists[i][j] = np.sqrt(np.sum(np.power(test_i - train_j, 2)))

        #dists[i][j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))

        #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

        #dist_sum += (X[i][j]**2 - self.X_train[i][j]**2)
        #print dists
      #break
      #dists[i][j] = np.sqrt(dist_sum)



      #pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      b = X[i,:]
      a = self.X_train
      dists[i,:] = np.sqrt(np.transpose(np.sum(np.power((b-a),2),axis=1)))
      #######################################################################
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    pass
    # 2ways
    #basis of the algorithm is (x-y)^2 = x^2 - 2*x*y + y^2
    #eg X- 3x7 and X_train-5x7
    #1st method
    # M = np.dot(X, self.X_train.T)#shape->3x5(resultant expected dim)
    # te = np.square(X).sum(axis = 1)#shape->(3x1)
    # tr = np.square(self.X_train).sum(axis = 1)#shape->(5x1)

    # #bringing the dimension that can be broadcasted on by using np.matrix or else doing transponse doesn't have a effect on te
    # dists = np.sqrt(-2*M+tr+np.matrix(te).T) #shape->(3x5 + 1x5 + 3x1) #bsically X is being broadcaster over the X_train matrix


    # #2nd method
    # x2 = np.sum(self.X_train*self.X_train, axis=1)
    # y2 = np.sum(X*X, axis=1)[None].T #convering normal array to matrix to get correct dimension
    # xy = np.dot(X, self.X_train.T)
    # dists = np.sqrt(x2 - 2*xy + y2)

    M = np.dot(X, self.X_train.T)
    te = np.square(X).sum(axis = 1)
    tr = np.square(self.X_train).sum(axis = 1)
    dists = np.sqrt(-2*M+tr+np.matrix(te).T)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      #Method1
      
      #print np.argsort(dists[i,:])
      #print "before flat",self.y_train[np.argsort(dists[i,:])]
      labels = self.y_train[np.argsort(dists[i,:])].flatten()
      #print "labels",labels,labels.shape
      closest_y = labels[0:k]
      #print "closest_y",closest_y

      #alternative
      # dists_sort = np.argsort(dists[i,:])
      # # print dists_sort
      # # print dists_sort < k
      # closest_y = self.y_train[np.where(dists_sort < k)].tolist()
      # print "closest_y",closest_y

      # #validating the correctness of above algo
      # state = closest_y
      # mask = np.in1d(self.y_train,state)
      # #print np.where(mask)
      # mask_arr = np.where(mask)[0]
      # assert self.y_train[mask_arr][0] in closest_y
      # assert self.y_train[mask_arr][1] in closest_y
      # assert self.y_train[mask_arr][2] in closest_y
      pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # Counter automatically breaks ties the right way (by choosing the smaller label):
      # >>> Counter([3, 2, 1, 3, 3, 3, 4, 1, 1, 1]).most_common(1)
      # [(1, 4)]
      c = Counter(closest_y)
      y_pred[i] = c.most_common(1)[0][0]


      #alternative
      # print type(closest_y),closest_y.count
      # y_pred[i] = max(set(closest_y), key=closest_y.count)
      #########################################################################
      #                           END OF YOUR CODE                            #
      #########################################################################

    return y_pred

  def get_accuracy(self,y_test_pred,y_test):
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / len(y_test)
    #print 'Got %d / %d correct => accuracy: %f %%' % (num_correct, num_test, accuracy*100.0)
    return accuracy
