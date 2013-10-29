from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import ModuleValidator
from pybrain.tools.validation import CrossValidator
from pybrain.structure.modules import *
import random

def main(numIn=22, datasource="PD",k=10,normalized=False,numEpochs=1000):
    norm='normalized_1_'if normalized else ''
    filename = "NN/" + str(datasource) + "_NN_" + norm+ str(numIn) + ".csv"
    dataFile = open(filename,'r')
    lines = [line.strip().split(',') for line in dataFile.readlines()]
    fullData = [[float(x) for x in n] for n in lines]

    random.shuffle(fullData)
    buckets = [fullData[i::k] for i in range(k)]

    wd = 0.01
    lrd = 1.0
    results = 0
    for wd in [0.05]:
        for l in [0.05]:#map(lambda x: x/100.0, range(80,100,10)):
            for m in [0.05]:#map(lambda x: x/100.0, range(0,100,1)):
                TP = TN = FP = FN = 0
                for i in range(k):
                    print i, ' fold'
                    testing = buckets[i]
                    training = buckets[0:i] + buckets[i+1:len(buckets)]
                    training = [datum for bucket in training for datum in bucket] #flatten

                    tr = ClassificationDataSet(numIn, 1, nb_classes=2)#, class_labels=["healthy","PD"])
                    ts = ClassificationDataSet(numIn, 1, nb_classes=2)#, class_labels=["healthy","PD"])

                    for data in training:
                        inputs = tuple(data[:-1])
                        output = data[-1:]
                        tr.addSample(inputs,output)

                    for data in testing:
                        inputs = tuple(data[:-1])
                        output = data[-1:]
                        ts.addSample(inputs,output)

                    tr._convertToOneOfMany()
                    ts._convertToOneOfMany()

                    expectedOutput = [int(out[0]) for out in ts.data['class'].tolist()]
                    print expectedOutput
                    
                    #add hiddenclass=LinearLayer
                    net = buildNetwork(tr.indim,13,tr.outdim,recurrent=True,hiddenclass=LinearLayer,outclass=SigmoidLayer,bias=True)
                    #return tr, ts, net, expectedOutput
                    trainer = BackpropTrainer(net,tr,learningrate=l,momentum=m)
                    #trainer.trainUntilConvergence(dataset=tr)
                    for j in range(numEpochs):
                        if j%1000 == 0: print j
                        trainer.train()

                    actualOutput = trainer.testOnClassData(ts)
                    print actualOutput
                    tp, tn, fp, fn = confusionMatrix(expectedOutput, actualOutput)
                    print tp,tn,fp,fn
                    TP += tp
                    TN += tn
                    FP += fp
                    FN += fn
                    print " TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
                    print
                    
                results = evaluate(TP,TN,FP,FN)
                print "learning: ", l, " momentum: ", m, " weight: ", wd
                if results['p'] > 0.5 and results['r'] > 0.0:
                    print "TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
                    print results
                    print


def evaluate(TP,TN,FP,FN):

    accuracy = precision = recall = 0

    if (TP+TN+FP+FN) > 0:
        accuracy = (TP+TN)/float(TP+TN+FP+FN)

    if (TP+FP) > 0:
        precision = TP/float(TP+FP)

    if (TP+FN) > 0:
        recall = TP/float(TP+FN)

    return {"a": accuracy, "p": precision, "r": recall}


def confusionMatrix(expected, actual):

    TP = TN = FP = FN = 0
    accuracy = precision = recall = 0

    pairs = zip(expected, actual)
    for pair in pairs:
        if pair == (1,1):
            TP += 1
        elif pair == (0,0):
            TN += 1
        elif pair == (0,1):
            FP += 1
        elif pair == (1,0):
            FN += 1

    return TP, TN, FP, FN
    
#    return results
#    ts, tr = dataset.splitWithProportion(0.10)
    
##    return expectedOutput, actualOutput

##    modval = ModuleValidator()
##    for i in range(100):
##          trainer.trainEpochs(1)
##          trainer.trainOnDataset(dataset=tr)
##          cv = CrossValidator( trainer, tr, n_folds=5, valfunc=modval.MSE )
##          print "MSE %f @ %i" %( cv.validate(), i )
##
##    print test
##    print ">", trainer.testOnClassData(dataset=ts)
##
##    return
##
##
#    evaluation = ModuleValidator.classificationPerformance(trainer.module,dataset)
##
##    evaluation = ModuleValidator.MSE(trainer.module,dataset)
##    validator = CrossValidator(trainer=trainer, dataset=dataset, n_folds=10, valfunc=evaluation)
##    return net,trainer, validator

##    trainer.trainOnDataset(dataset,100) #do 1000
##    trainer.testOnData(verbose=True)
##
##
##    print "Number of training patterns: ", len(train)
##    print "Input and output dimensions: ", train.indim, train.outdim
##    print "First sample (input, target, class):"
##    #print train['input'][0], train['target'][0], train['class'][0]
##    return
    #return (test,train)        
