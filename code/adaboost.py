import numpy.core.multiarray
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import random
import dataparser
import sys
import numpy as np
import PDneuralnets

#baselearner = decision tree classifier
class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def runAdaboost(dataset_name="Parkinsons",learners=1000):
    d = dataparser.DataSet(name=dataset_name)
    examples = [example[1:] for example in d.examples]
    n_samples = len(examples)
    n_features = len(examples[0])-1
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=np.int)

    for i, ir in enumerate(examples):
        data[i] = np.asarray(ir[:-1], dtype=np.float)
        target[i] = np.asarray(ir[-1], dtype=np.int)

    dataset = Bunch(data=data, target=target)

    weak_learner = SVC(C=10000,kernel='poly',degree=1,probability=True)
    
#    clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=learners)
    clf = AdaBoostClassifier(n_estimators=learners)
    
    inputsList = dataset.data.tolist()
    outputsList = dataset.target.tolist()

    for i, inputs in enumerate(inputsList):
        inputs.append(outputsList[i])

    k=10
    random.shuffle(inputsList)
    buckets = [inputsList[i::k] for i in range(k)]

    TP = TN = FP = FN = 0
    for i in range(k):
        testing = buckets[i]
        training = buckets[0:i] + buckets[i+1:len(buckets)]
        training = [datum for bucket in training for datum in bucket] #flatten

        trIn = [x[:-1] for x in training]
        trOut = [x[-1] for x in training]

        tsIn = [x[:-1] for x in testing]
        expected = [x[-1] for x in testing]

        clf.fit(trIn, trOut)

        actual = clf.predict(tsIn)

        tp, tn, fp, fn = PDneuralnets.confusionMatrix(expected, actual)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

    print TP,TN,FP,FN
    results = PDneuralnets.evaluate(TP,TN,FP,FN)
    print results

if __name__ == '__main__':
    numLearners = 1000
    if len(sys.argv) == 2:
        numLearners = int(sys.argv[1])

    runAdaboost(learners=numLearners)



