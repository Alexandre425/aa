import numpy as np
from matplotlib import pyplot as plt

# Generates the polinomial X matrix for a training set x and the order
def X_matrix(x, order):
    return np.array([[x[l]**p for p in range(order+1)] for l in range(np.shape(x)[0])])

# Calculates a list of predictions of a polinomial model based on a list of input data
def polinomialModel(x, Beta):
    x.sort()
    return np.array([np.sum([Beta[i] * _x**i for i in range(len(Beta))]) for _x in x])

# Calculates the sum of squared errors of the model, and the SSE of the outliers
def SSE(X, y, Beta):
    sse =  np.linalg.norm(y-np.dot(X, Beta))**2
    avg_se = sse / X.shape[0]
    sse_in = 0
    for i in range(X.shape[0]):
        line_sse = abs(y[i][0] - np.dot(X[i], Beta))**2
        if (line_sse > avg_se * 8): # If the squared error of the sample is far from the average
            sse_in += line_sse      # Means it's an outlier, add it to the outlier SSE
    
    return sse, sse_in
    


# Loads two data files, fits a model with a certain order and plots the result
def loadFitAndPlot(x_file, y_file, order):
    x = np.load(x_file)
    x_list = [_[0] for _ in x]   # Convert to list of values
    y = np.load(y_file)
    # Workaround to one of the files being (50,) instead of (50,1)
    if (len(y.shape) == 1):
        y = np.array([[_y] for _y in y])
    X = X_matrix(x_list, order)
    # Calculate the parameters through the normal equation
    Beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    Beta_list = [_[0] for _ in Beta]
    sse, sse_outliers = SSE(X, y, Beta)
    # Print and plot
    if (sse_outliers > 0):
        print(f"Coefficients: {Beta.T} SSE: {sse} Inlier-only SSE: {sse - sse_outliers}") 
    else:
        print(f"Coefficients: {Beta.T} SSE: {sse}") 
    plt.scatter(x, y)
    plt.plot(x_list, polinomialModel(x_list, Beta_list))
    plt.title("Data from " + x_file)
    plt.show()

if __name__ == "__main__":
    ### Exercise 2.1
    
    ## 1)

    # First determining the SSE of the model:
    #   SSE = ||y - XB||^2
    #
    # Where the matrix X is given by:
    #   X = [[1 x_1 x_1^2 ... x_1^p]
    #        ...
    #        [1 x_n x_n^2 ... x_n^p]]
    #
    # Where n is the number of samples in the training set
    #   and p is the order of the polinomial
    #
    # Then, by calculating the gradient of the SSE and equaling it to zero
    #   we can obtain the normal equation:
    #   (X^T X)B = X^T y
    #
    # From which we can obtain:
    #   B = (X^T X)^-1 X^T y
    
    ## 3)
    loadFitAndPlot("data/data1_x.npy", "data/data1_y.npy", 1)

    ## 4)
    loadFitAndPlot("data/data2_x.npy", "data/data2_y.npy", 2)
    # The model fits the data quite accurately. It also ignores the noise effectively
    #   because of the low order of the polynomial

    ## 5)
    loadFitAndPlot("data/data2a_x.npy", "data/data2a_y.npy", 2)
    # The model fits the data with decent accuracy
    # The inlier-only SSE is significantly larger than the one of the previous case.
    #   Seeing as both of them have the same data with the exception of the outliers,
    #   we can infer that the high sensitivity of the LS method to outliers was what
    #   caused the increase in SSE for the inliers. 
    # It is also possible to view the effect of the outliers through the plot. 
    #   The model follows the part of the wave with a negative slope
    #   with a significant error, the curve being above the inliers significantly.
    #   In layman's terms, it can be said that the two outliers above the wave "pulled"
    #   the model closer to them and away from the inliers.
    #   
    # It can also be explained mathematically, due to the fact the method gives greater
    #   "importance" to outliers, as the error of those particular data points is high,
    #   which when squared becomes even higher, and thus have more influence on the model.

    ### 2.2
    ## 2)
    # Ridge regression is a form of regression which has a regularization term, which
    #   penalizes coeficients with large values, a way of preventing overfitting and
    #   selecting for relevant features. It is simmilar to the LS method, being given 
    #   by the following expression:
    #   min(SSE + lambda||B||^2)
    #
    # The first term being the SSE, and the second the regularization term. A large 
    #   coefficient vector leads to a large result, which is contrary to the objective,
    #   minimizing it.
    #
    # Lasso regression is simmilar, but uses a different regularization term, given by:
    #   lambda||B||_1^2
    #   where the norm is the l1 norm, a simple sum of the absolute values of the coefficients.
    #
    # This forces the sum of the coefficients to be below a certain value, directly related to
    #   lambda, which then forces some of the coefficients to be zero if lambda is sufficiently
    #   large. It is superior at feature selection when compared to ridge regression,
    #   as it reduces the value of the coefficients equally and independently of
    #   their value, while the former has "diminishing returns" for small coefficients,
    #   meaning it will reduce the length of the vector in general, but generally doesn't
    #   eliminate coefficients completely, and as such doesn't select for features as well.
    #


    loadFitAndPlot("data/data3_x.npy", "data/data3_y.npy", 1)