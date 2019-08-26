import sys
import getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def usage():
    print('python hand_gestures_deceit_detector.py -i <input file> -o <output file>')

def predictNPlot(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    yPred = model.predict(X_test)
    print("Accuracy score:%s" % accuracy_score(y_test, yPred))
    print("Classification report:")
    print("********************************************************")
    print(classification_report(y_test, yPred))
    print("********************************************************\n")
    skplt.metrics.plot_confusion_matrix(y_test, yPred, normalize=True)
    plt.title('Confusion Matrix')
    plt.show()
    return yPred


def main(argv):
    inFile = None
    outFile = None

    try:
        opts, args = getopt.getopt(argv,"hi:o:")
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-i':
            inFile = arg
        elif opt == '-o':
            outFile = arg
        else:
            usage()
            print('Invalid argument %s' % opt)
            sys.exit(2)

    if (None == inFile) or (None == outFile):
        print("Missing arguments")
        usage()
        sys.exit(2)

    posturesData = pd.read_csv(inFile)
    posturesData.drop(0,inplace=True)
    posturesData.reset_index(inplace=True,drop=True)

    #View a plot of the Class counts
    sns.countplot(x="Class", data=posturesData)
    plt.show()

    #View a plot of the User counts
    sns.countplot(x="User", data=posturesData)
    plt.show()

    classLabels = posturesData['Class']
    users = posturesData['User']
    posturesData.drop(columns = ['Class', 'User'], inplace=True)
    posturesData.replace('?',0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(posturesData, classLabels, test_size=0.2)

    print('RANDOM FOREST:')
    clfRF = RandomForestClassifier(n_estimators=100,random_state=0)
    yPred = predictNPlot(clfRF, X_train, y_train, X_test, y_test)
    with open(outFile, 'w') as f:
        f.write("Random Forest prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    print('DECISION TREE:')
    clfDT = tree.DecisionTreeClassifier()
    yPred = predictNPlot(clfDT, X_train, y_train, X_test, y_test)
    with open(outFile, 'a') as f:
        f.write("Decision Tree prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    print('LOGISTIC REGRESSION:')
    clfLR = LogisticRegression(C=1.0, random_state=0, solver='sag', multi_class='ovr')
    yPred = predictNPlot(clfLR, X_train, y_train, X_test, y_test)
    with open(outFile, 'a') as f:
        f.write("Logistic Regression prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    print('MLP:')
    clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    yPred = predictNPlot(clfMLP, X_train, y_train, X_test, y_test)
    with open(outFile, 'a') as f:
        f.write("MLP prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    height = [0.987,0.96, 0.79, 0.53]
    bars = ('Random Forest', 'Decision Tree', 'Logistic regression', 'MLP')
    y_pos = np.arange(len(bars))
    plt.title("Performance Comparison")
    plt.bar(y_pos, height, color=['cyan', 'red', 'green', 'blue'])
    plt.xticks(y_pos, bars)
    plt.show()

main(sys.argv[1:])
