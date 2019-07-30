from __future__ import division
import sys
import glob
import os
import collections
import json
import string
import re
import math

classlists = ['positive_polaritydeceptive_from_MTurk', 'positive_polaritytruthful_from_TripAdvisor',
              'negative_polaritydeceptive_from_MTurk', 'negative_polaritytruthful_from_Web']
priorsNC = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
allclasses = ['positive', 'negative', 'truthful', 'deceptive']


def readfiles(all_files):
    # defaultdict is analogous to dict() [or {}], except that for keys that do not
    # yet exist (i.e. first time access), the value gets contructed using the function
    # pointer (in this case, list() i.e. initializing all keys to empty lists).
    test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
    train_by_class = collections.defaultdict(list)

    test_by_class_for_analysis = {}
    sizeofall = 0

    for f in all_files:
        # Take only last 4 components of the path. The earlier components are useless
        # as they contain path to the classes directories.
        class1, class2, fold, fname = f.split('/')[-4:]
        if fold == 'fold1':
            # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
            test_by_class[class1 + class2].append(f)
            test_by_class_for_analysis[f] = [class1, class2]
        else:
            train_by_class[class1 + class2].append(f)
        # train_by_class[class1 + class2].append(f)

    for key, val in priorsNC.items():
        for classlist in classlists:
            if key in classlist:
                priorsNC[key] += len(train_by_class[classlist])

    for classlist in classlists:
        sizeofall += len(train_by_class[classlist])

    allvalidwords_test = None
    allvalidwordscounts_test = None
    allvalidwords_train, allvalidwordscounts_train, allvalidwordscountsdocwc = extractwords(train_by_class, priorsNC, sizeofall)

    wdpn, biaspn = learn_perceptron(sizeofall, allvalidwordscountsdocwc, allvalidwords_train, allvalidwordscounts_train, 'positive')
    wdtd, biastd = learn_perceptron(sizeofall, allvalidwordscountsdocwc, allvalidwords_train, allvalidwordscounts_train, 'truthful')

    with open('vanillamodel.txt', 'w') as vanillamodel_w:
        for term in allvalidwords_train:
            vanillamodel_w.write(term + '\n')
        vanillamodel_w.write('\n\nend of allvalidwords***************\n')
        vanillamodel_w.write('sizeofall: {}\n'.format(sizeofall))
        vanillamodel_w.write('\n\nstart of word counts***************\n\n')
        vanillamodel_w.write(json.dumps(allvalidwordscounts_train, indent=2))
        vanillamodel_w.write('\n\nstart of file word count ***************\n\n')
        vanillamodel_w.write(json.dumps(allvalidwordscountsdocwc, indent=2))
        vanillamodel_w.write('\n\nstart of wd weight pn***************\n\n')
        vanillamodel_w.write(json.dumps(wdpn, indent=2))
        vanillamodel_w.write('\n\nstart of wd weight td***************\n\n')
        vanillamodel_w.write(json.dumps(wdtd, indent=2))
        vanillamodel_w.write('\n\nstart of biaspn***************\n\n')
        vanillamodel_w.write(str(biaspn))
        vanillamodel_w.write('\n\nstart of biastd***************\n\n')
        vanillamodel_w.write(str(biastd))


    wdpnave, biaspnave = learn_perceptron_average(sizeofall, allvalidwordscountsdocwc, allvalidwords_train, allvalidwordscounts_train,
                                    'positive')
    wdtdave, biastdave = learn_perceptron_average(sizeofall, allvalidwordscountsdocwc, allvalidwords_train, allvalidwordscounts_train,
                                    'truthful')
    with open('averagedmodel.txt', 'w') as averagedmodel_w:
        for term in allvalidwords_train:
            averagedmodel_w.write(term + '\n')
        averagedmodel_w.write('\n\nend of allvalidwords***************\n')
        averagedmodel_w.write('sizeofall: {}\n'.format(sizeofall))
        averagedmodel_w.write('\n\nstart of word counts***************\n\n')
        averagedmodel_w.write(json.dumps(allvalidwordscounts_train, indent=2))
        averagedmodel_w.write('\n\nstart of file word count ***************\n\n')
        averagedmodel_w.write(json.dumps(allvalidwordscountsdocwc, indent=2))
        averagedmodel_w.write('\n\nstart of wd weight pn***************\n\n')
        averagedmodel_w.write(json.dumps(wdpnave, indent=2))
        averagedmodel_w.write('\n\nstart of wd weight td***************\n\n')
        averagedmodel_w.write(json.dumps(wdtdave, indent=2))
        averagedmodel_w.write('\n\nstart of biaspn***************\n\n')
        averagedmodel_w.write(str(biaspnave))
        averagedmodel_w.write('\n\nstart of biastd***************\n\n')
        averagedmodel_w.write(str(biastdave))


    filesign = {}

    classify(wdpn, biaspn, test_by_class, 'positive', filesign)
    classify(wdtd, biastd, test_by_class, 'truthful', filesign)


    return [all_files, test_by_class, train_by_class, allvalidwords_test, allvalidwordscounts_test, allvalidwords_train,
            allvalidwordscounts_train, priorsNC, sizeofall, filesign, test_by_class_for_analysis]


def extractstopwords():
    with open('./stopwords.txt') as stopwordsfile:
        stopwords = stopwordsfile.readline()
    return stopwords


def extractwords(group_to_test, priorsNC, sizeofall):
    stopwords = extractstopwords()
    allvalidwords = []
    allvalidwordscounts = {'positive_polaritydeceptive_from_MTurk': {},
                           'positive_polaritytruthful_from_TripAdvisor': {},
                           'negative_polaritydeceptive_from_MTurk': {},
                           'negative_polaritytruthful_from_Web': {}}
    allvalidwordscountsdocwc = {} #allvalidwordscountsdocwc docwc is doc word count

    for key, value in group_to_test.items():
        for f in value:
            allvalidwordscountsdocwc[f] = {}
            with open(f) as filetoread:
                stringtoprocess = filetoread.readline()
                pun = re.sub(r'[^\w\s]', '', stringtoprocess)
                pun = pun.lower()
                tokens = pun.split(' ')
                for token in tokens:
                    if token not in stopwords:
                        token = token.replace("\n", "")
                        if token not in allvalidwords and token != '':
                            allvalidwords.append(token)
                        if token != '':
                            try:
                                allvalidwordscounts[key][token] += 1
                                allvalidwordscountsdocwc[f][token] += 1
                            except:
                                allvalidwordscounts[key][token] = 1
                                allvalidwordscountsdocwc[f][token] = 1

    return [allvalidwords, allvalidwordscounts, allvalidwordscountsdocwc]


def learn_perceptron(sizeofall, docwordcount, allvalidwords_train, allvalidwordscounts_train, phase):
    wd = {word:0 for word in allvalidwords_train}
    bias = 0
    for iteration in range(100):
        for file, val in docwordcount.items():
            activation = 0
            ylabel = 1 if(phase in file) else -1
            for word, count in val.items():
                activation += wd[word]*count
            activation += bias
            if (ylabel*activation <=0):
                for word, count in val.items():
                    wd[word] += ylabel * count
                bias += ylabel

    return [wd, bias]

def learn_perceptron_average(sizeofall, docwordcount, allvalidwords_train, allvalidwordscounts_train, phase):
    wd = {word:0 for word in allvalidwords_train}
    ucache = {word:0 for word in allvalidwords_train}
    bias, beta, cinc = [0, 0, 1]
    for iteration in range(100):
        for file, val in docwordcount.items():
            activation = 0
            ylabel = 1 if(phase in file) else -1
            for word, count in val.items():
                activation += wd[word]*count
            activation += bias
            if (ylabel*activation <=0):
                for word, count in val.items():
                    wd[word] += ylabel * count
                    ucache[word] += ylabel * count * cinc
                bias += ylabel
                beta += ylabel * cinc
            cinc += 1

    for word in wd.keys():
        wd[word] -= (ucache[word]/cinc)
    bias -= beta/cinc

    return [wd, bias]


def classify(wd, bias, group_to_test, phase, filesign):

    allvalidwords_test, allvalidwordscounts_test = extractwords_for_test(group_to_test)


    for file, val in allvalidwordscounts_test.items():
        activation = 0
        for word, count in val.items():
            if word not in wd:
                continue
            activation += wd[word] * count
        activation += bias
        if file not in filesign:
            filesign[file] = []
        if activation >= 0:
            if phase=='positive':
                filesign[file].append('positive')
            if phase=='truthful':
                filesign[file].append('truthful')
        if activation < 0:
            if phase=='positive':
                filesign[file].append('negative')
            if phase=='truthful':
                filesign[file].append('deceptive')
    return

def extractwords_for_test(group_to_test):  # test or development
    stopwords = extractstopwords()
    allvalidwords = []
    allvalidwordscounts = {}

    for key, value in group_to_test.items():
        for f in value:
            with open(f) as filetoread:
                stringtoprocess = filetoread.readline()
                pun = re.sub(r'[^\w\s]', '', stringtoprocess)
                pun = pun.lower()
                tokens = pun.split(' ')
                allvalidwordscounts[f] = {}

                for token in tokens:
                    if token not in stopwords:
                        token = token.replace("\n", "")
                        if token not in allvalidwords:
                            allvalidwords.append(token)
                        if token != '':
                            try:
                                allvalidwordscounts[f][token] += 1
                            except:
                                allvalidwordscounts[f][token] = 1
    # print(json.dumps(allvalidwordscounts, indent=2))

    return [allvalidwords, allvalidwordscounts]


def compute_multinomial(allvalidwords_test, allvalidwordscounts_test, allvalidwords_train, allvalidwordscounts_train,
                        priorsNC, sizeofall):
    multinomresult = {}
    sizeofdict = len(allvalidwords_train)

    allwordsnum = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}

    for classnum in allclasses:
        for classlist in classlists:
            if classnum in classlist:
                for word, val in allvalidwordscounts_train[classlist].items():
                        allwordsnum[classnum] += val

    for file, val in allvalidwordscounts_test.items():
        multinomresult[file] = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
        for classtype in allclasses:
            for word, count in val.items():
                sumwords = 0
                for classlist in classlists:
                    if classtype in classlist:
                        try:
                            sumwords += allvalidwordscounts_train[classlist][word]
                        except:
                            pass
                multinomresult[file][classtype] += (math.log((sumwords + 1) / (allwordsnum[classtype] + sizeofdict)) * count)
                # print('&&&&&&&&&&&&&&&&&&&&&&&')
                # print(math.log((sumwords + 1) / (allwordsnum[classtype] + sizeofdict)))
                # print(sumwords)
                # print(word)
                # print((sumwords + 1) / (allwordsnum[classtype] + sizeofdict))
                # print('&&&&&&&&&&&&&&&&&&&&&&&')

            multinomresult[file][classtype] += math.log(priorsNC[classtype] / sizeofall)

    outputtags = {}
    comparebetweenclass = [['positive', 'negative'], ['truthful', 'deceptive']]
    for file, val in multinomresult.items():
        outputtags[file] = []
        for xx in range(2):
            maxresult = multinomresult[file][comparebetweenclass[xx][0]]

            if maxresult < multinomresult[file][comparebetweenclass[xx][1]]:
                outputtags[file].append(comparebetweenclass[xx][1])
            else:
                outputtags[file].append(comparebetweenclass[xx][0])

    return outputtags

def analysis(outputtags, test_by_class_for_analysis):
    tps = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    fps = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    fns = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    precision = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    recall = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    f1 = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}

    for file, value in outputtags.items():
        if value[0] == 'positive' and 'positive' in test_by_class_for_analysis[file][0]:
            tps['positive'] += 1
        if value[0] == 'positive' and 'negative' in test_by_class_for_analysis[file][0]:
            fps['positive'] += 1
        if value[0] == 'negative' and 'positive' in test_by_class_for_analysis[file][0]:
            fns['positive'] += 1

        if value[0] == 'negative' and 'negative' in test_by_class_for_analysis[file][0]:
            tps['negative'] += 1
        if value[0] == 'negative' and 'positive' in test_by_class_for_analysis[file][0]:
            fps['negative'] += 1
        if value[0] == 'positive' and 'negative' in test_by_class_for_analysis[file][0]:
            fns['negative'] += 1

        if value[1] == 'truthful' and 'truthful' in test_by_class_for_analysis[file][1]:
            tps['truthful'] += 1
        if value[1] == 'truthful' and 'deceptive' in test_by_class_for_analysis[file][1]:
            fps['truthful'] += 1
        if value[1] == 'deceptive' and 'truthful' in test_by_class_for_analysis[file][1]:
            fns['truthful'] += 1

        if value[1] == 'deceptive' and 'deceptive' in test_by_class_for_analysis[file][1]:
            tps['deceptive'] += 1
        if value[1] == 'deceptive' and 'truthful' in test_by_class_for_analysis[file][1]:
            fps['deceptive'] += 1
        if value[1] == 'truthful' and 'deceptive' in test_by_class_for_analysis[file][1]:
            fns['deceptive'] += 1

    for iterator in precision.keys():
        try:
            precision[iterator] = tps[iterator] / (tps[iterator] + fps[iterator])
        except:
            # print(iterator)
            pass
    print(tps)
    print(fps)
    print(fns)
    for iterator in recall.keys():
        recall[iterator] = tps[iterator] / (tps[iterator] + fns[iterator])

    for iterator in f1.keys():
        f1[iterator] = (2 * (precision[iterator] * recall[iterator])) / (precision[iterator] + recall[iterator])
    print(precision)
    print(recall)
    print(f1)
    pass


if __name__ == "__main__":
    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    all_files, test_by_class, train_by_class, allvalidwords_test, \
    allvalidwordscounts_test, allvalidwords_train, allvalidwordscounts_train, priorsNC, sizeofall, filesign, test_by_class_for_analysis = readfiles(all_files)
    # outputtags = compute_multinomial(allvalidwords_test, allvalidwordscounts_test, allvalidwords_train, allvalidwordscounts_train, priorsNC, sizeofall)
    analysis(filesign, test_by_class_for_analysis)