import sys
sys.path.append('source/')

import numpy.core.multiarray
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_svmlight_file
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

#    weak_learner = SVC(C=10000,kernel='poly',degree=1,probability=True)
    
 #   clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=learners)
    
    inputsList = dataset.data.tolist()
    outputsList = dataset.target.tolist()

    for i, inputs in enumerate(inputsList):
        inputs.append(outputsList[i])

    k=10
    random.shuffle(inputsList)
    buckets = [inputsList[i::k] for i in range(k)]

    TP = TN = FP = FN = 0
    for i in range(k):
        clf = AdaBoostClassifier(n_estimators=learners)
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
    return

def sparseAdaboost(svmfilepath="combinedsvm.dat",learners=100,factor=1):
    sparsedata, target = load_svmlight_file(svmfilepath)

    data = sparsedata.toarray()
    
    dataset = Bunch(data=data, target=target)
    
    inputsList = dataset.data.tolist()
    outputsList = dataset.target.tolist()

    posExamples = []
    negExamples = []
    
    for i, inputs in enumerate(inputsList):
        inputs.append(outputsList[i])
        if outputsList[i] == 1:
            posExamples.append(inputs)            
        elif outputsList[i] == -1:
            negExamples.append(inputs)        

    trainSets,testSets = splitData(posExamples,negExamples,factor=factor)
    
    sum1 = sum([example[-1] for example in trainSets[0] if example[-1] == 1])

    print 'oversampling by factor of', factor
    print sum1, ' positive ', len(trainSets[0]) - sum1, ' negative in trainsets'
  
    TP = TN = FP = FN = 0
    for i in range(3):
        clf = AdaBoostClassifier(n_estimators=learners)
        testing = testSets[i]
        training = trainSets[i]

        trIn = [x[:-1] for x in training]
        trOut = [x[-1] for x in training]

        tsIn = [x[:-1] for x in testing]
        expected = [x[-1] for x in testing]
        
        clf.fit(trIn,trOut)

        actual = clf.predict(tsIn)
        actual = actual.tolist()

        tp, tn, fp, fn = PDneuralnets.confusionMatrix(expected, actual,neg=-1.0)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

    print TP,TN,FP,FN
    results = PDneuralnets.evaluate(TP,TN,FP,FN)
    recall = results['r']
    precision = results['p']
    f = (2 * recall * precision) / (recall + precision)
    print "F SCORE", f
    print results
    return

def combinedAdaboost(dataset_name="Parkinsons",learners=100,factor=180):
    d = dataparser.DataSet(name=dataset_name)
#    examples = [example[1:] for example in d.examples]

    examples1, examples2 = combineDatasets(d, dataparser.PDregression)

    posExamples = [example for example in examples1 if example[-1] == 1]
    negExamples = [example for example in examples1 if example[-1] == 0]

    random.shuffle(examples2)

    trainSets,testSets = splitData(posExamples,negExamples,factor=factor,additional=examples2)

    print 'oversampling by factor of', factor
 
    sum1 = sum([example[-1] for example in trainSets[0]])
 
    print sum1, ' positive ', len(trainSets[0]) - sum1, ' negative in trainsets'
     
 #  weak_learner = SVC(C=10000,kernel='poly',degree=1,probability=True)
    
 #   clf = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=learners)
    clf = AdaBoostClassifier(n_estimators=learners)

    TP = TN = FP = FN = 0
    for i in range(3):
 
        clf = AdaBoostClassifier(n_estimators=learners)
    
        training = trainSets[i]
        testing = testSets[i]

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
    recall = results['r']
    precision = results['p']
    f = (2 * recall * precision) / (recall + precision)
    print "F SCORE", f
    print results
    return


def combineDatasets(ds1,ds2):
    attrs1 = set(ds1.attrnames)
    attrs2 = set(ds2.attrnames)
    commonAttrs = attrs1 & attrs2
    indicesDs1 = [ds1.attrnames.index(attrname) for attrname in commonAttrs]
    indicesDs2 = [ds2.attrnames.index(attrname) for attrname in commonAttrs]
    
    
    examplesDs1 = [[example[i] for i in indicesDs1] + [example[ds1.target]] for example in ds1.examples]
    examplesDs2 = [[example[i] for i in indicesDs2] + [1] for example in ds2.examples]
    #regression target is 1

    return examplesDs1, examplesDs2

    

def splitData(posExamples,negExamples,factor=1,additional=[]):
    testSet1 = posExamples[:16] + negExamples[:16]
    testSet2 = posExamples[16:32] + negExamples[16:32]
    testSet3 = posExamples[32:48] + negExamples[32:48]

    trainSet1 = posExamples[16:] + negExamples[16:]*factor + additional
    trainSet2 = posExamples[:16] + posExamples[32:] + (negExamples[:16] + negExamples[32:])*factor + additional
    trainSet3 = posExamples[:32] + posExamples[48:] + (negExamples[:32] + negExamples[48:])*factor + additional

    random.shuffle(testSet1)
    random.shuffle(testSet2)
    random.shuffle(testSet3)

    random.shuffle(trainSet1)
    random.shuffle(trainSet2)
    random.shuffle(trainSet3)
    
    trainSets = [trainSet1, trainSet2, trainSet3]
    testSets = [testSet1, testSet2, testSet3]

    return trainSets,testSets


if __name__ == '__main__':
    numLearners = 1000
    if len(sys.argv) == 2:
        numLearners = int(sys.argv[1])

#    runAdaboost(learners=numLearners)
#    combinedAdaboost(learners=numLearners)


