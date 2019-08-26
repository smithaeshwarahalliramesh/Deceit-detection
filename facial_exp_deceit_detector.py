import sys
import getopt
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

def usage():
    print('python facial_exp_deceit_detector.py -i <training data> -t <test data> -o <output prediction file>')
   
def metricNPlot(model, X_test, y_test, yPred):
    print("Accuracy score:%s" % model.score(X_test, y_test))
    print("Classification report:")
    print("********************************************************")
    print(classification_report(y_test, yPred))
    print("********************************************************")
    skplt.metrics.plot_confusion_matrix(y_test, yPred, normalize=True)
    plt.title('Confusion Matrix')
    plt.show()
    fpr, tpr, thresholds = roc_curve(y_test, yPred)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    _ = plt.xlabel('False Positive Rate')
    _ = plt.ylabel('True Positive Rate')
    _ = plt.title('Label propagation ROC Curve')
    _ = plt.xlim([-0.02, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower right")
    plt.show()

def main(argv):
    trainFile = None
    testFile = None
    outFile = None

    try:
        opts, args = getopt.getopt(argv,"hi:t:o:")
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-i':
            trainFile = arg
        elif opt == '-t':
            testFile = arg
        elif opt == '-o':
            outFile = arg
        else:
            usage()
            print('Invalid argument %s' % opt)
            sys.exit(2)

    if (None == trainFile) or (None == testFile) or (None == outFile):
        print("Missing arguments")
        usage()
        sys.exit(2)

    facialData = pd.read_csv(trainFile)
    testData = pd.read_csv(testFile)

    testData.drop(columns=['id'], inplace=True)
    testData.reset_index(inplace=True, drop=True)

    labels = testData['class']
    classLabels = []
    for i in range(len(labels)):
        classLabels.append(1 if (labels[i] == 'deceptive') else 0)
    testData.drop(columns = ['class'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(testData, classLabels, test_size=0.2, stratify=classLabels, random_state=42)

    X_train.insert(1, "class", y_train) 
    sns.countplot(x="class", data=X_train)
    X_train = X_train.drop(columns=['class'])

    # Label Propagation
    modelLabelProp = LabelPropagation()
    labels = [-1] * len(facialData[:10000])
    labels.extend(y_train)
    inputData = pd.concat([facialData[:10000], X_train], sort=False, ignore_index=True, copy=False)
    modelLabelProp.fit(inputData, labels)
    yPred = modelLabelProp.predict(X_test)
    print("LABEL PROPAGATION:")
    metricNPlot(modelLabelProp, X_test, y_test, yPred)
    
    with open(outFile, 'w') as f:
        f.write("Label Propagation prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    # Label Spreading
    modelLabelSpread = LabelSpreading(kernel='knn', n_neighbors=15)
    labels = [-1] * len(facialData[:10000])
    labels.extend(y_train)
    inputData = pd.concat([facialData[:10000], X_train], sort=False, ignore_index=True, copy=False)
    modelLabelSpread.fit(inputData, labels)
    yPred = modelLabelSpread.predict(X_test)
    print("LABEL SPREADING:")
    metricNPlot(modelLabelSpread, X_test, y_test, yPred)

    with open(outFile, 'a') as f:
        f.write("Label Spreading prediction\n")
        for item in yPred:
            f.write("%s\n" % item)

    height = [0.8, 0.68]
    bars = ('Label Propagation', 'Label Spreading')
    y_pos = np.arange(len(bars))
    plt.title("Performance Comparison")
    plt.bar(y_pos, height, color=['cyan', 'red'])
    plt.xticks(y_pos, bars)
    plt.show()

main(sys.argv[1:])
