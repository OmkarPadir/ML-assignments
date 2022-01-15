# Name: Omkar Pramod Padir
# Student Id: 20310203
# Dataset id:9-18-9
# Course: Machine Learning CS7CS4


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# function to plot given data
def PlotBaselineData():
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')

    plt.scatter(X1, X2, 50, y, cmap=cmap)


    plt.legend(handles=[red_patch, green_patch])

cmap, norm = mcolors.from_levels_and_colors([-1, 0, 1], ['red', 'green'])
red_patch = mpatches.Patch(color='red', label='-1 (data)')
green_patch = mpatches.Patch(color='green', label='1 (data)')

# Part A starts here

# Load data and create arrays of input and output

df = pd.read_csv("ML_W2_DATA.csv")

X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

# Plot the baseline data

PlotBaselineData()
plt.show()

# Create and fit Logistic Regression Model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X,y)

# Model Parameters and score

print("Logistic Regression intercept: "+str(lr.intercept_))
print("Logistic Regression coefficients: "+str(lr.coef_))
print("Logistic Regression score: "+str(lr.score(X,y)))

PlotBaselineData()
cmap2, norm = mcolors.from_levels_and_colors([-1,0,1], ['blue', 'yellow'])
blue_patch = mpatches.Patch(color='blue', label='-1 (prediction)')
yellow_patch = mpatches.Patch(color='yellow', label='1 (prediction)')
plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch])
plt.scatter(X1, X2, 10, lr.predict(X), cmap=cmap2, marker="o")

# Solve the equation to get x2 values by using coefficient and intercept of the model

point1= -(lr.intercept_[0]+(lr.coef_[0][0]))/(lr.coef_[0][1]) #x2 value when x1 is +1
point2= -(lr.intercept_[0]-(lr.coef_[0][0]))/(lr.coef_[0][1]) #x2 value when x1 is -1

plt.plot([1,-1],[point1,point2],'tab:cyan')
plt.show()


# Part B starts here


from sklearn.svm import LinearSVC

# Train SVM models for different values of C

lsvc_001=LinearSVC(C=0.001).fit(X,y)
lsvc_point1=LinearSVC(C=0.1).fit(X,y)
lsvc_1=LinearSVC(C=1).fit(X,y)
lsvc_100=LinearSVC(C=100).fit(X,y)
lsvc_1000=LinearSVC(C=1000).fit(X,y)

svc_arr=[lsvc_001,lsvc_point1,lsvc_1,lsvc_100,lsvc_1000]
svc_name=['SVM, C=0.001','SVM, C=0.1','SVM, C=1','SVM, C=100','SVM, C=1000']

for i in range(5):
    print("For model "+svc_name[i])
    print("Intercept: " + str(svc_arr[i].intercept_))
    print("Coefficients: " + str(svc_arr[i].coef_))
    print("Score: " + str(svc_arr[i].score(X,y)))

# function to plot prediction data points of the model
def PlotSVMData(model):
    plt.scatter(X1, X2, 10, model.predict(X), cmap=cmap2, marker="o")
    plt.legend(handles=[red_patch,green_patch,blue_patch, yellow_patch])

# function to plot decision boundary of the model
def PlotLine(lr):
    point1 = -(lr.intercept_[0] + (lr.coef_[0][0])) / (lr.coef_[0][1])  # x2 value when x1 is +1
    point2 = -(lr.intercept_[0] - (lr.coef_[0][0])) / (lr.coef_[0][1])  # x2 value when x1 is -1

    plt.plot([1, -1], [point1, point2], 'tab:cyan')

# plot all svm models
for m in svc_arr:
    PlotBaselineData()
    PlotSVMData(m)
    PlotLine(m)
    plt.show()


# Part C starts here

X1_sq=np.square(df.iloc[:,0]) # Square of first parameter
X2_sq=np.square(df.iloc[:,1]) # Square of second parameter

X_inputs=np.column_stack((X1,X2,X1_sq,X2_sq))

# Train the Logistic Regression
lr_sq = LogisticRegression(penalty='none')
lr_sq.fit(X_inputs,y)

print("Squared Logistic Regression intercept: "+str(lr_sq.intercept_))
print("Squared Logistic Regression coefficients: "+str(lr_sq.coef_))
print("Squared Logistic Regression score: "+str(lr_sq.score(X_inputs,y)))

PlotBaselineData()
plt.scatter(X1, X2, 10, lr_sq.predict(X_inputs), cmap=cmap2, marker="o")
plt.legend(handles=[ red_patch,green_patch,blue_patch, yellow_patch])

# plot decision boundary
x1a = np.linspace(-0.75,0.75,100)   # Random x1 values from -0.75 to 0.75

# comparing with a*x*x + b*x + c = 0
a=lr_sq.coef_[0][3]
b=lr_sq.coef_[0][1]

x2a = []
x2b = []

# find values of c and solve for x2
for k in x1a:
    c=( (lr_sq.coef_[0][0]*k) + (lr_sq.coef_[0][2]*k*k))
    tt =np.absolute(((b*b) - (4*a*c)))
    root1 = (-b + np.sqrt(tt))/(2*a)
    x2a.append(root1)

plt.plot(x1a,x2a)
plt.show()

# Plot of x1*x1 against x2. Just for reference
plt.xlabel('Feature1 sqaured')
plt.ylabel('Feature2')
plt.scatter(X1_sq, X2, 10, y, cmap=cmap)
plt.legend(handles=[red_patch,green_patch])
plt.show()

# Comparison with baseline predictor
count_p1=0
count_m1=0
total=0
for t in y:

    if(t==-1):
        count_m1=count_m1+1
    else:
        count_p1=count_p1+1
    total=total+1

print(count_p1 , count_m1)
if count_p1 > count_m1:
    print("Accuracy of baseline predictor: "+str(count_p1/total))
else:
    print("Accuracy of baseline predictor: " + str(count_m1/total))