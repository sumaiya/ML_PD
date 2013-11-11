import sys
sys.path.append('source/')

from utils import *
import agents, random, operator
import csv
import adaboost

#______________________________________________________________________________

class DataSet:
    """A data set for a machine learning problem.  It has the following fields:

    d.examples    A list of examples (including both inputs and outputs).  
                  Each one is a list of attribute values.
    d.attrs       A list of integers to index into an example, so example[attr]
                  gives a value. Normally the same as range(len(d.examples)). 
    d.attrnames   Optional list of mnemonic names for corresponding attrs 
                  (including both inputs and outputs).
    d.target      The index of the attribute that a learning algorithm will try 
                  to predict. By default the final attribute.
    d.targetname  The name of the attribute that the learning algorithm will try
                  to predict.
    d.inputs      The list of attrs (indices, not names) without the target (in 
                  other words, a list of the indices of input attributes).
    d.values      A list of lists, each sublist is the set of possible
                  values for the corresponding attribute (including both inputs
                  and outputs). If None, it is computed from the known examples 
                  by self.setproblem. If not None, an erroneous value raises 
                  ValueError.
    d.name        Name of the data set (for output display only).
    d.source      URL or other source where the data came from.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, target=-1, targetName="total", values=None,
                 name='',
                 inputs=None, exclude=(), doc=''):
        """Accepts any of DataSet's fields.  Examples can also be a string 
        or file from which to parse examples using parse_csv.
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        update(self, name=name, values=values)

        #Assume that the examples are stored in a file named 'name'.csv.
        self.examples = parse_csv(DataFile(name+'.csv').read())
        if name == "ParkinsonsRegression":
            if targetName == "motor":
                measure = 0
            elif targetName == "total":
                measure = 1
            for example in self.examples:
                targets = example[4:6] #motor or total UPDRS
                example[:] = example[:4] + example[6:len(example)]
                example += [targets[measure]]
        elif name == "Parkinsons" or "parkinsons_balanced":
            for example in self.examples:
                targets = example[17:18] #status
                example[:] = example[:17] + example[18:len(example)]
                example += targets

        self.attrnames = self.examples[0]
        self.examples = self.examples[1:]

        #check to make sure that the examples have been read in properly,
        #  mostly just that their values are in the right range for each
        #  attribute
        map(self.check_example, self.examples)
        
        # Attrs are the indices of examples, unless otherwise stated.
        attrs = range(len(self.examples[0]))
        self.attrs = attrs
            
        self.setproblem(target, inputs=inputs, exclude=exclude)
        
        #grab the output attribute name, just for printing
        self.targetname = self.attrnames[self.target]

    def __str__(self):
        """Returns a string representation of the DataSet object."""
        s = "The " + str(self.name) + " dataset contains the following " + str(len(self.inputs)) + " input attributes (followed by their possible values):\n" 
        for input in self.inputs:
            s += str(self.attrnames[input]) + "\t" + str(self.values[input]) + "\n"
        s += "The output to be predicted is \"" + str(self.attrnames[self.target]) + "\" with possible values:\n" 
        s += str(self.values[self.target]) + ".\n"    
        s += "The dataset contains " + str(len(self.examples)) + " training examples.  Here is one example:\n" 
        s += str(self.examples[0])
        return s

    def setproblem(self, target, inputs=None, exclude=()):
        """Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not put use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""
        self.target = self.attrnum(target)
        exclude = map(self.attrnum, exclude)
        if inputs:
            self.inputs = removall(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs
                           if a is not self.target and a not in exclude]
        if not self.values:
            self.values = map(unique, zip(*self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value %s for attribute %s in %s' %
                                     (example[a], self.attrnames[a], example))

    def attrnum(self, attr):
        "Returns the number used for attr, which can be a name, or -n .. n."
        if attr < 0:
            return len(self.attrs) + attr
        elif isinstance(attr, str): 
            return self.attrnames.index(attr)
        else:
            return attr

    def sanitize(self, example):
       "Return a copy of example, with non-input attributes replaced by 0."
       return [i in self.inputs and example[i] for i in range(len(example))] 

    def __repr__(self):
        return '<DataSet(%s): %d examples, %d attributes>' % (
            self.name, len(self.examples), len(self.attrs))

#______________________________________________________________________________

def parse_csv(input, delim=','):
    """Input is a string consisting of lines, each line has comma-delimited 
    fields.  Convert this into a list of lists.  Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    #Here is the code from AIMA - below it is a more verbose version....
    #lines = [line for line in input.splitlines() if line.strip() is not '']
    #return [map(num_or_str, line.split(delim)) for line in lines]

    #separate the input into a list of rawlines (only non-empty ones)
    rawlines = []
    for line in input.splitlines():
        if not line.strip() == '':
            rawlines.append(line.strip())
 
    #split each line into a list of cells and turn cells into nums where appropriate
    lines = []
    for line in rawlines:
        cells = line.split(delim)
        cells = map(num_or_str, cells)
        lines.append(cells)
        
    return lines

#______________________________________________________________________________

def uci_to_svm(dataset, examples, startAt=1, suffix="SVM.dat", path="",attrNum=22):
    """Using a Dataset object, create a file of data in SVMLight format"""
    filename = path + dataset.name + suffix
    f = open(filename, 'w')
    target = dataset.target
    #start at 1 to ignore subject ID attr
    attrs = [x for x in dataset.attrs[startAt:] if x is not target]
    four_attrs_to_keep = [16,17,18,22]
    ten_attrs_to_keep = [5,8,13,14,15,16,17,18,21,22]
    if attrNum==10:
        attrs = ten_attrs_to_keep
    elif attrNum==4:
        attrs = four_attrs_to_keep
    for example in examples:
        if example[target] == 1:
            output = "+1"
        elif example[target] == 0:
            output = "-1"
        line = output + " "
        for attr in attrs:
            line += str(attr) + ":" + str(example[attr]) + " "
        f.write(line + "\n")
    f.close()
    return


# sparse_svm_format("combinedsvm.dat", PD, PDregression)

def sparse_svm_format(filename,dataset1,dataset2):
    examples1, examples2 = adaboost.combineDatasets(dataset1,dataset2)

    target = len(examples1[0])-1
    attrs = range(len(examples1[0])-1)

    PDdata = DataSet(name="Parkinsons")

    toAppend = [1,2,3,19,20]

    for i, PDexample in enumerate(PDdata.examples):
        for at in toAppend:
            examples1[i].append(PDexample[at])

    examples = examples1 + examples2

    shortLength = len(attrs)
    appending = range(shortLength,shortLength+len(toAppend))

    attrs = range(1,len(examples[0])+1)
    attrs.remove(len(examples2[0]))
        
    f = open(filename,'w')
    for example in examples:
        if len(example) == shortLength + 1:
            output = "-1"
        elif example[target] == 1:
            output = "+1"
        elif example[target] == 0:
            output = "-0"
        line = output + " "
        for attr in attrs:
            if attr <= len(example):
                line += str(attr) + ":" + str(example[attr-1]) + " "
        f.write(line + "\n")
    f.close()
    return
#______________________________________________________________________________

def uci_to_svm_k_fold(dataset, startAt=1, k=10, path="PDexamples/",attrNum=22):
    """ Generate train and test files for k fold cross validation """
    examples = dataset.examples
    random.shuffle(examples)
    buckets = [examples[i::k] for i in range(k)]

    for i in range(k):
        testing = buckets[i]
        training = buckets[0:i] + buckets[i+1:len(buckets)]
        training = [datum for bucket in training for datum in bucket] #flatten
        uci_to_svm(dataset, training, startAt, "_train_" + str(i) + "_SVM.dat", path,attrNum)
        uci_to_svm(dataset, testing, startAt, "_test_" + str(i) + "_SVM.dat", path,attrNum)

    return
#___________________________________________________________________

def dataset_to_csv(dataset, filename, useAttrs=22, normalized=True):
    attrs = dataset.attrs[1:]
    if useAttrs == 4:
        attrs = [16,17,18,22,23]
    elif useAttrs == 10:
        attrs = [5,8,13,14,15,16,17,18,21,22,23]

    if normalized:
        examplesToUse = normalizeData(dataset)
    else:
        examplesToUse = dataset.examples

    examples = [[example[i] for i in attrs] for example in examplesToUse]
    
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(examples)
    
    return

def makeMaxMinDict(dataset):
    ranges = {}
    for attribute in dataset.attrs:
        values = [x[attribute] for x in dataset.examples]
        ranges[attribute] = (min(values), max(values))
    return ranges

def normalizeData(dataset):
    ranges = makeMaxMinDict(dataset)
    ymax = 1
    ymin = 0

    normalizedExamples = []
    for example in dataset.examples:
        normalizedExample = [example[0]]
        for attribute in dataset.attrs[1:-1]:
            xmin, xmax = ranges[attribute]
            y = ((ymax-ymin) * (int(example[attribute])-xmin) / (xmax-xmin)) + ymin
            normalizedExample.append(y)
        normalizedExample.append(example[dataset.attrs[-1]])
        normalizedExamples.append(normalizedExample)

    return normalizedExamples

PD = DataSet(name="Parkinsons")
balanced = DataSet(name="parkinsons_balanced")
PDregression = DataSet(name="ParkinsonsRegression")
    
