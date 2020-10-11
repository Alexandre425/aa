import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model as linmod
from sklearn import metrics as met


# Generates the polinomial X matrix for a training set x and the order
def X_matrixPoly(x, order):
    return np.array([[x[l]**p for p in range(order+1)] for l in range(np.shape(x)[0])])

def X_matrixLinear(x):
    return np.array([np.array([1] + [_x for _x in x[l]]) for l in range(np.shape(x)[0])])

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
    X = X_matrixPoly(x_list, order)
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

def loadRidgeAndLasso(x_file, y_file):
    x = np.load(x_file)
    y = np.load(y_file)
    a = np.linspace(1e-3, 10, 1000)

    least_squares = linmod.LinearRegression()
    least_squares.fit(x, y)
    ls_coefs = np.array(least_squares.coef_)

    coefs = []
    for _a in a:
        ridge = linmod.Ridge(alpha=_a, max_iter=10000)
        ridge.fit(x, y)
        coefs.append(ridge.coef_[0])
    ridge_coefs = np.array(coefs)

    coefs = []
    for _a in a:
        lasso = linmod.Lasso(alpha=_a, max_iter=10000)
        lasso.fit(x, y)
        coefs.append(lasso.coef_)
    lasso_coefs = np.array(coefs)


    [plt.hlines(c, 0, 10) for c in ls_coefs]
    [plt.scatter(a, ridge_coefs[:,c], label=f"Coef {c+1}") for c in range(ridge_coefs.shape[1])]
    plt.xscale("log")
    plt.xlim(1e-3, 10)
    plt.title("Ridge regression")
    plt.xlabel("a")
    plt.ylabel("Coefficients")
    plt.legend()
    plt.show()

    [plt.hlines(c, 0, 10) for c in ls_coefs]
    [plt.scatter(a, lasso_coefs[:,c], label=f"Coef {c+1}") for c in range(lasso_coefs.shape[1])]
    plt.xscale("log")
    plt.xlim(1e-3, 10)
    plt.title("Lasso regression")
    plt.xlabel("a")
    plt.ylabel("Coefficients")
    plt.legend()
    plt.show() 

    a = 0.071   # first lambda that removes the irrelevant coefficient altogether
    lasso = linmod.Lasso(alpha=a, max_iter=10000)
    lasso.fit(x, y)
    y_pred = lasso.predict(x)
    y_pred_ls = least_squares.predict(x)
    # Sorting both according to one so the data is readable
    y_list = [y[i][0] for i in range(y.shape[0])]
    yp_list = [y_pred[i] for i in range(y_pred.shape[0])]
    yp_list_ls = [y_pred_ls[i] for i in range(y_pred_ls.shape[0])]
    y_list, yp_list, yp_list_ls = zip(*sorted(zip(y_list, yp_list, yp_list_ls)))
    y = np.array(y_list)
    y_pred = np.array(yp_list)
    y_pred_ls = np.array(yp_list_ls)
    print(f"Lasso SSE: {met.mean_squared_error(y, y_pred)*y.shape[0]}")
    print(f"LS SSE: {met.mean_squared_error(y, y_pred_ls)*y.shape[0]}")
    plt.scatter(range(y.shape[0]), y, label="Training Data")
    plt.scatter(range(y_pred.shape[0]), y_pred, label="Lasso prediction")
    plt.scatter(range(y_pred_ls.shape[0]), y_pred_ls, label="LS prediction")
    plt.title("Lasso model prediction")
    plt.xlabel("Data point")
    plt.ylabel("Prediction")
    plt.legend()
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
    
    loadRidgeAndLasso("data/data3_x.npy", "data/data3_y.npy")

    ## 6)
    # The results match the what was previously stated, with lasso regression quickly 
    #   selecting for the relevant features. The first feature whose coefficient
    #   was nullified first was the second, meaning it is the irrelevant feature.
    # Another thing to note is that the coefficients of the two methods are identical
    #   to the coefficients of the least squares method when lambda (or a) is zero,
    #   which makes sense from a mathematical standpoint, as it nullifies the second term
    #   of the minimizer, making it identical to the sum of squared errors.

    ## 7)
    # The chosen value for lambda/alpha is 0.071, which is the first value shown to 
    #   nullify the irrelevant term completely. 
    # The lasso method has a slightly larger squared error, which is natural seeing as
    #   how the alpha also influences the other coefficients, making them smaller, but also
    #   making the model less accurate as a result. Still, the difference in SSE is negligible,
    #   especially when compared to the computational power the lasso method saves when
    #   predicting. Seeing as how one of the coefficients was completely nullified, the
    #   processing power required to compute the prediction goes down by one third, since
    #   one of the three features is ignored. This would have a significant advantage if the 
    #   model had to be applied a large number of times, like is often the case.
