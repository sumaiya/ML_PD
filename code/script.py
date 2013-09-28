#!/usr/bin/env python
import argparse
from subprocess import call
import os
import dataparser
import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', '-k', default="10")
    p.add_argument('--cost', '-j', default="1")
    p.add_argument('--algorithm', '-a', default="SVM")
    options = p.parse_args()
    
    k = int(options.fold)
    j = options.cost
    data_name = "parkinsons_balanced"
    PD = dataparser.DataSet(name=data_name)
    identifier = str(k) + "_fold_" + str(options.cost) + "_cost_" + datetime.datetime.now().strftime("%H%M%S")
    path = "PDexamples/" + identifier + "/"
    os.system("mkdir " + str(path))

    dataparser.uci_to_svm_k_fold(PD,1,k,path)

    os.chdir("svm_light/")
    path = "../" + path


    for j in range(1,100,10):#map(lambda x: x/float(10), range(1,100, 10)):
        TP = TN = FP = FN = 0
        for i in range(k):
            train = path + data_name + "_train_" + str(i) + "_SVM.dat"
            test = path + data_name + "_test_" + str(i) + "_SVM.dat"
            model = path + data_name + "_model_" + str(i) + str(j) + "_SVM.dat"
            predictions = path + data_name + "_predictions_" + str(i) + str(j) + "_SVM.dat"
            learn = "./svm_learn" + " -t 1 -d 2 -c " + str(j) + " " + train + " " + model + " > ../silence.txt"
            os.system(learn)
            
            classify = "./svm_classify " + test + " " + model + " " + predictions + " > ../silence.txt"
            os.system(classify)
            
            expectations = [line[0] for line in open(test)]
            results = []
            for line in open(predictions):
                if line[0] == "-":
                    results += "-"
                else:
                    results += "+"

            for i in range(len(results)):
                if expectations[i] == "+":
                    if results[i] == "+":
                        TP += 1
                    else:
                        FN += 1
                elif expectations[i] == "-":
                    if results[i] == "-":
                        TN += 1
                    else:
                        FP += 1

##        f = open(path + "results.txt", "w")
##        f.write("TP = "+str(TP) + "\n")
##        f.write("TN = "+str(TN) + "\n")
##        f.write("FP = "+str(FP) + "\n")
##        f.write("FN = "+str(FN) + "\n")
##        f.write("Accuracy = "+str((TP+TN)/float(TP+TN+FP+FN)) + "\n")
##        f.write("Precision = "+str(TP/float(TP+FP)) + "\n")
##        f.write("Recall = "+str(TP/float(TP+FN)) + "\n")
##        f.write("Total = "+str(TP+TN+FP+FN) + "\n")
##        f.close()
        print "Cost factor is " + str(j)
        print "TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
        print "Accuracy = "+str((TP+TN)/float(TP+TN+FP+FN))
        print "Precision = "+str(TP/float(TP+FP))
        print "Recall = "+str(TP/float(TP+FN))
        print "Total = "+str(TP+TN+FP+FN) + "\n"

##        results = open(output)
##        # assumes equal size buckets...
##        for line in results:
##            if "Accuracy on test set" in line:
##                accuracy += float(line[len("Accuracy on test set: "):line.find("%")])
##                total += int(line[line.find("incorrect, ") + len("incorrect, "):line.rfind(" total")])
##            if "Precision/recall" in line:
##                precision += float(line[len("Precision/recall on test set: "):line.find("%")])
##                recall += float(line[line.find("%/")+len("%/"):line.rfind("%")])
##    
##    accuracy = accuracy / k
##    precision = precision / k
##    recall = recall / k
##
##    f = open(path+"results.txt", "w")
##    f.write("Accuracy = "+str(accuracy) + "\n")
##    f.write("Precision = "+str(precision) + "\n")
##    f.write("Recall = "+str(recall) + "\n")
##    f.write("Total = "+str(total) + "\n")
##    f.close()

if __name__ == '__main__':
    main()
