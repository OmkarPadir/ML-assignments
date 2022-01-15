# Name: Omkar Pramod Padir
# Student Id: 20310203
# Dataset id:4-4-4
# Course: Machine Learning CS7CS4
# Week 3 Assignment


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# Part 1-A starts here

# Load data and create arrays of input and output

df = pd.read_csv("ML_W3_DATA.csv")

X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X1,X2,y, c='red')

# Used to add color legends in the plots
# Code referred from: https://www.geeksforgeeks.org/how-to-manually-add-a-legend-with-a-color-box-on-a-matplotlib-figure/

red_patch = mpatches.Patch(color='red', label='(Given Data)')
yellow_patch = mpatches.Patch(color='lightyellow', label='(Model Predictions)')

# add labels and display
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Target')
plt.legend(handles=[red_patch])
plt.title('Scatter plot of given dataset')
plt.show()

# Part 1-B starts here

from sklearn.preprocessing import PolynomialFeatures

# this function will provide all combinations of inputs to degrree given in parameters
# eg. degree 3 for inputs a and b will give 1, a, b, a^2, a*b, b^2, a^3, (a^2)*b , a*(b^2), b^3.
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

poly = PolynomialFeatures(5)
X = poly.fit_transform(X)


from sklearn import linear_model
from sklearn.model_selection import train_test_split
C_Values = [1, 5, 10, 50, 100, 500]         # Different C values for changing penalty strength
lasso_array=[]

for Ci in C_Values:
    lasso_reg = linear_model.Lasso(alpha=1/(2*Ci))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lasso_reg.fit(X_train,y_train)
    lasso_array.append(lasso_reg)

    print("\nC value: "+str(Ci))
    print("Lasso coeff: "+str(lasso_reg.coef_))
    print("Lasso score: "+str(lasso_reg.score(X_test,y_test)))  # Shows the accuracy of the model


# 1-C starts here

# Create grid of inputs ranging from -2 to 2
X_test =[]
grid=np.linspace(-2,2)

for i in grid:
    for j in grid:
        X_test.append([i,j])

X_test = np.array(X_test)

# Use polynomialfeatures on the newly created test data.

poly = PolynomialFeatures(5)
X_test_poly = poly.fit_transform(X_test)

# Plot all predictions and training data for all models
i=0
for l in lasso_array:

    y_pred=l.predict(X_test_poly)

    #https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    cmap, norm = mcolors.from_levels_and_colors([-1, 0, 1], ['lightyellow', 'lightyellow'])

    surf = ax.plot_trisurf(X_test[:,0],X_test[:,1],y_pred, cmap=cmap, alpha=0.6)

    ax.scatter(X1,X2,y,c='red')

    ax.set_xlabel('Feature1')
    ax.set_ylabel('Feature2')
    ax.set_zlabel('Target')
    plt.legend(handles=[red_patch,yellow_patch],loc=2)
    plt.title('Lasso Regression C='+str(C_Values[i]),loc='right')
    plt.show()
    i=i+1


# 1-e Ridge Regression
C_Values_ridge=[0.0001,0.001,0.01,0.1,1]
Ridge_array=[]
for Ci in C_Values_ridge:
    ridge_reg = linear_model.Ridge(alpha=1/(2*Ci))

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
    ridge_reg.fit(X_train1,y_train1)
    Ridge_array.append(ridge_reg)

    print("\nC value: "+str(Ci))
    print("Ridge coeff: "+str(ridge_reg.coef_))
    print("Ridge score: "+str(ridge_reg.score(X_test1,y_test1)))

i=0
# Plot all predictions and training data for all models
for r in Ridge_array:

    y_pred=r.predict(X_test_poly)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.

    cmap, norm = mcolors.from_levels_and_colors([-1, 0, 1], ['lightyellow', 'lightyellow'])

    surf = ax.plot_trisurf(X_test[:,0],X_test[:,1],y_pred, cmap=cmap, alpha=0.6)

    ax.scatter(X1,X2,y,c='red')

    ax.set_xlabel('Feature1')
    ax.set_ylabel('Feature2')
    ax.set_zlabel('Target')
    plt.legend(handles=[red_patch,yellow_patch],loc=2)
    plt.title('Ridge Regression C='+str(C_Values_ridge[i]),loc='right')
    plt.show()
    i=i+1


# Part 2 Lasso Regression K-fold Cross Validation

mean_error=[]; std_error=[]

#Loop for all models
for Ci in C_Values:
    lasso_reg = linear_model.Lasso(alpha=1/(2*Ci))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    temp = []

    # Loop for all k-fold splits
    for train, test in kf.split(X):
        lasso_reg.fit(X[train],y[train])
        ypred = lasso_reg.predict(X[test])
        print("\nC value: "+str(Ci))
        print("Lasso coeff: "+str(lasso_reg.coef_))

        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

# Plot the error bar for Lasso Regression
import matplotlib.pyplot as plt
plt.errorbar(C_Values,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Mean square error')
plt.xlim((0.1,100))
plt.title('Lasso Regression Mean Square Error for Different C')
plt.show()


# Part 2 for Ridge Regression K-fold Cross Validation

mean_error=[]; std_error=[]
#Loop for all models
for Ci in C_Values_ridge:
    ridge_reg = linear_model.Ridge(alpha=1/(2*Ci))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    temp = []

    # Loop for all k-fold splits
    for train, test in kf.split(X):
        ridge_reg.fit(X[train],y[train])
        ypred = ridge_reg.predict(X[test])
        print("\nC value: "+str(Ci))
        print("Ridge coeff: "+str(ridge_reg.coef_))

        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

# Plot the error bar for Ridge Regression
import matplotlib.pyplot as plt
plt.errorbar(C_Values_ridge,mean_error,yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Mean square error')
plt.xlim((0.0001,0.1))
plt.title('Ridge Regression Mean Square Error for Different C')
plt.show()
