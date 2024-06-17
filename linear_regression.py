# Nicholas Moreland
# 1001886051

import numpy as np

# Train our linear regression model
def training_stage(training_file, degree, lambda1):
    # Load training data
    train_file = open(training_file)
    input_arr = np.loadtxt(train_file, dtype=float)

    train_columns = len(input_arr[0])
    M = ((train_columns - 1) * degree) + 1

    I = np.identity(M, dtype=float)

    input_arr = np.hsplit(input_arr, [train_columns - 1, train_columns - 1]);
    train_arr = input_arr[0]
    target_arr = input_arr[2]
    phi_matrix = np.ones(shape=(len(train_arr), M))

    # Create phi matrix
    for i in range(len(train_arr)):
        temp = 1
        for j in range(train_columns - 1):
            phi_matrix[i][temp:(temp+degree)] = train_arr[i][j]
            temp = temp + degree 

    # Create phi matrix for degree > 1
    if degree > 1:
        for i in range(len(train_arr)):
            for j in range(1,M):
                temp = ((j-1) % degree) + 1
                phi_matrix[i][j]= (phi_matrix[i][j]) ** temp
    
    # Find w using the formula
    w = np.dot(lambda1, I) + np.dot(phi_matrix.T, phi_matrix)
    w = np.linalg.pinv(w)
    w = np.dot(w, phi_matrix.T)
    w = np.dot(w, target_arr)

    i = 0
    np.set_printoptions(precision=4)
    for x in w:
        print("w%d=%.4f" % (i, x))
        i += 1
    
    return w

# Test our linear regression model
def testing_stage(testing_file, w, degree):
    # Load testing data
    test_file = open(testing_file)
    input_arr = np.loadtxt(test_file, dtype=float)

    test_columns = len(input_arr[0])
    M = ((test_columns - 1) * degree) + 1

    # Identity matrix
    I = np.identity(M, dtype=float)

    input_arr = np.hsplit(input_arr, [test_columns - 1, test_columns - 1]);
    test_arr = input_arr[0]
    target_arr = input_arr[2]
    phi_matrix = np.ones(shape=(len(test_arr), M))

    # Create phi matrix
    for i in range(len(test_arr)):
        temp = 1
        for j in range(test_columns - 1):
            phi_matrix[i][temp:(temp+degree)] = test_arr[i][j]
            temp = temp + degree 

    # Create phi matrix for degree > 1
    if degree > 1:
        for i in range(len(test_arr)):
            for j in range(1,M):
                temp = ((j-1) % degree) + 1
                phi_matrix[i][j]= (phi_matrix[i][j]) ** temp
    
    # Linear regression model from the training data
    y = np.dot(w.T, phi_matrix.T)

    # Calculate the mean squared error
    new_list = []
    target_list = target_arr.tolist()

    # Create a new list of target values
    for i in range(len(target_list)):
        new_list.append(target_list[i].pop())
    
    # Calculate the mean squared error
    error_list = []
    for j in range(len(y[0])):
        error_list.append((new_list[j] - y[0][j]) ** 2)

    # Print statement
    for j in range(len(y[0])):
        print("ID=%5d, output=%14.4f, target value = %.4f, squared error = %.4f" % (j+1, y[0][j], new_list[j], error_list[j]))

# Train our linear regression model
def linear_regression(training_file, test_file, degree, lambda1):
    # Run training stage model
    w = training_stage(training_file, degree, lambda1)

    # Run testing stage model
    testing_stage(test_file, w, degree)

