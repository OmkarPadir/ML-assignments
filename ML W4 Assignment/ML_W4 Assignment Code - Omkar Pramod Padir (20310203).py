# Name: Omkar Pramod Padir
# Student Id: 20310203
# Dataset 1 id: 19--38-19-0
# Dataset 2 id: 19--19-19-0
# Course: Machine Learning CS7CS4
# Week 4 Assignment


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# Part 1 starts here

# Load data and create arrays of input and output

# Dataset

DatasetArray=["W4_D1.csv","W4_D2.csv"]  # Same analysis for both datasets

for d in DatasetArray:

    # Load Data
    df = pd.read_csv(d)

    X1=df.iloc[:,0]
    X2=df.iloc[:,1]
    X=np.column_stack((X1,X2))
    y=df.iloc[:,2]

    # Colors for plots and legends
    cmap, norm = mcolors.from_levels_and_colors([-1, 0, 1], ['red', 'green'])
    red_patch = mpatches.Patch(color='red', label='-1 (data)')
    green_patch = mpatches.Patch(color='green', label='1 (data)')
    cmap2, norm2 = mcolors.from_levels_and_colors([-1, 0, 1], ['blue', 'yellow'])
    blue_patch = mpatches.Patch(color='blue', label='-1 (prediction)')
    yellow_patch = mpatches.Patch(color='yellow', label='1 (prediction)')
    bluecmap, norm3 = mcolors.from_levels_and_colors([-1, 0, 1], ['blue', 'blue'])
    yellowcmap, norm4 = mcolors.from_levels_and_colors([-1, 0, 1], ['yellow', 'yellow'])

    # function to plot given data
    def PlotBaselineData():
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')

        plt.scatter(X1, X2, 50, y, cmap=cmap)

        plt.legend(handles=[red_patch, green_patch])


    plt.title("Plot of Training Data")
    PlotBaselineData()
    plt.show()

    # Function to plot predictions

    def PlotPredictionData(Prediction):

        # If all cases are +1 use yellow
        if (np.count_nonzero(Prediction==-1) == 0):
            plt.scatter(X1, X2, 10, Prediction, cmap=yellowcmap, marker="o")
        else:
            plt.scatter(X1, X2, 10, Prediction, cmap=cmap2, marker="o")


    from sklearn.preprocessing import PolynomialFeatures

    # this function will provide all combinations of inputs to degrree given in parameters
    # eg. degree 3 for inputs a and b will give 1, a, b, a^2, a*b, b^2, a^3, (a^2)*b , a*(b^2), b^3.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html


    from sklearn import linear_model
    from sklearn.neighbors import KNeighborsClassifier

    D_Values = [1,2,3,4,5]            # Different Degree values
    C_Values = [0.1, 1, 10, 100]      # Different C values for changing penalty strength
    K_Values = [1,3,5,7,9]            # Different K values for KNN

    mean_score=[]
    std_error=[]

    # Check performance of all Degrees
    for Di in D_Values:
        Xi=[]
        poly = PolynomialFeatures(Di)
        Xi = poly.fit_transform(X)

        model = linear_model.LogisticRegression(penalty='l2',solver='lbfgs')

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        temp = []

        # Loop for all k-fold splits
        for train, test in kf.split(Xi):
            model.fit(Xi[train],y[train])
            ypred = model.predict(Xi[test])

            from sklearn.metrics import f1_score
            temp.append(f1_score(y[test],ypred))

        mean_score.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        # Plot Baseline and Predictions
        PlotBaselineData()
        PlotPredictionData(model.predict(Xi))
        plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch])
        plt.title("Predictions for Degree = "+str(Di))
        plt.show()

    # Plot F1 scores
    plt.errorbar(D_Values,mean_score,yerr=std_error,linewidth=2)
    plt.xlabel('Di'); plt.ylabel('F1 Score')
    plt.title("F1 Score for different Degrees")
    plt.show()

    # Check performance for different C values:
    mean_score=[]
    std_error=[]

    for Ci in C_Values:
        Xi=[]
        poly = PolynomialFeatures(2)    # 2 chosen from f1 score analysis above
        Xi = poly.fit_transform(X)

        model = linear_model.LogisticRegression(penalty='l2',C=Ci,solver='lbfgs')
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        temp = []

        # Loop for all k-fold splits
        for train, test in kf.split(Xi):
            model.fit(Xi[train],y[train])
            ypred = model.predict(Xi[test])

            from sklearn.metrics import f1_score
            temp.append(f1_score(y[test],ypred))

        mean_score.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        # Plot Baseline and Predictions
        PlotBaselineData()
        PlotPredictionData(model.predict(Xi))
        plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch])
        plt.title("Predictions for C= "+str(Ci))
        plt.show()

    # Plot F1 score
    import matplotlib.pyplot as plt
    plt.errorbar(C_Values,mean_score,yerr=std_error,linewidth=2)
    plt.xlabel('Ci'); plt.ylabel('F1 Score')
    plt.xlim((0.05,11))
    plt.title("F1 Score for different C Values")
    plt.show()

    # Loop for different K values:
    mean_score=[]
    std_error=[]

    for Ki in K_Values:
        Xi=X

        model = KNeighborsClassifier(n_neighbors=Ki)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        temp = []

        # Loop for all k-fold splits
        for train, test in kf.split(Xi):
            model.fit(Xi[train],y[train])
            ypred = model.predict(Xi[test])

            from sklearn.metrics import f1_score
            temp.append(f1_score(y[test],ypred))

        mean_score.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        PlotBaselineData()
        PlotPredictionData(model.predict(Xi))
        plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch])
        plt.title("KNN Predictions for K= "+str(Ki))
        plt.show()



    plt.errorbar(K_Values,mean_score,yerr=std_error,linewidth=2)
    plt.xlabel('Ki'); plt.ylabel('F1 Score')
    plt.title("F1 Score for different K Values in KNN")
    plt.show()

    # Confusion Matrix of finalized models

    from sklearn.model_selection import train_test_split

    poly = PolynomialFeatures(2)    # 2 chosen from f1 score analysis above for Logistic Regression
    Xi = poly.fit_transform(X)

    LR_X_train, LR_X_test, LR_y_train, LR_y_test = train_test_split(Xi, y, test_size=0.2, random_state=42)
    KNN_X_train, KNN_X_test, KNN_y_train, KNN_y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Finalize LR model where C=1
    LR_Final = linear_model.LogisticRegression(penalty='l2',C=1,solver='lbfgs').fit(LR_X_train,LR_y_train)
    # Finalize KNN model where K=2
    KNN_Final = KNeighborsClassifier(n_neighbors=3).fit(KNN_X_train,KNN_y_train)


    from sklearn.dummy import DummyClassifier
    dummy_clf = DummyClassifier(strategy="most_frequent")   # Dummy classifier that predicts most frequent class
    dummy_clf.fit(KNN_X_train, KNN_y_train)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve

    print("Confusion Matrix for Final LR Model")
    print(confusion_matrix(LR_y_test, LR_Final.predict(LR_X_test)))

    print("Confusion Matrix for Final KNN Model")
    print(confusion_matrix(KNN_y_test, KNN_Final.predict(KNN_X_test)))

    print("Confusion Matrix for Dummy Model")
    # Using KNN split because it represents input without augmented features i.e. Polyfeatures() NOT applied
    print(confusion_matrix(KNN_y_test, dummy_clf.predict(KNN_X_test)))


    # Roc Curve for all finalized models

    LR_fpr, LR_tpr, _ = roc_curve(LR_y_test,LR_Final.decision_function(LR_X_test))
    plt.plot(LR_fpr,LR_tpr, color= 'orange')

    KNN_fpr, KNN_tpr, _ = roc_curve(KNN_y_test,KNN_Final.predict_proba(KNN_X_test)[:,1])
    plt.plot(KNN_fpr,KNN_tpr, color='blue')

    D_fpr, D_tpr, _ = roc_curve(KNN_y_test,dummy_clf.predict_proba(KNN_X_test)[:,1])
    plt.plot(D_fpr,D_tpr , color='green',linestyle='dashed')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    orange_roc = mpatches.Patch(color='orange', label='Logistic Regression')
    blue_roc = mpatches.Patch(color='blue', label='KNN')
    green_roc = mpatches.Patch(color='green', label='Dummy model')
    plt.legend(handles=[orange_roc,blue_roc,green_roc])
    plt.title('ROC Curve for different models')
    plt.show()