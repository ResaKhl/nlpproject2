from __future__ import division
import sys
import glob
import os
import collections
import json
import string
import re
import math
import ast

classlists = ['positive_polaritydeceptive_from_MTurk', 'positive_polaritytruthful_from_TripAdvisor',
              'negative_polaritydeceptive_from_MTurk', 'negative_polaritytruthful_from_Web']
priorsNC = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
allclasses = ['positive', 'negative', 'truthful', 'deceptive']


def readfiles(all_files, modeltoread):
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
        test_by_class[class1 + class2].append(f)
        test_by_class_for_analysis[f] = [class1, class2]

    allvalidwords_train = []
    allvalidwordscounts_train = {} #not used

    with open(modeltoread, 'r') as train_file:
        for line in train_file:
            if ('end of allvalidwords' in line):
                break
            token = line.replace("\n", "")
            if token != '':
                allvalidwords_train.append(token)
        temp = ''
        for line in train_file:
            if ('start of word counts' in line):
                break
            temp += line
        tokensize = temp.split(' ')
        sizeofall = int(tokensize[1].replace('\n', ''))
        allstring = ''
        for line in train_file:
            if ('start of file word count' in line):
                break
            allstring += line.replace("\n", "")
        allvalidwordscounts_train = ast.literal_eval(allstring)
        allstring = ''
        for line in train_file:
            if ('start of wd weight pn' in line):
                break
            allstring += line.replace("\n", "")
        docwordcount = ast.literal_eval(allstring)
        allstring = ''
        for line in train_file:
            if ('start of wd weight td' in line):
                break
            allstring += line.replace("\n", "")
        wdpn = ast.literal_eval(allstring)
        allstring = ''
        for line in train_file:
            if ('start of biaspn' in line):
                break
            allstring += line.replace("\n", "")
        wdtd = ast.literal_eval(allstring)
        allstring = ''
        for line in train_file:
            if ('start of biastd' in line):
                break
            allstring += line.replace("\n", "")
        biaspn = float(allstring.split(' ')[0])
        allstring = ''
        for line in train_file:
            allstring += line.replace("\n", "")
        biastd = float(allstring.split(' ')[0])

        print(json.dumps(wdpn, indent=2))
        print('\n\n\n\dudul\n\n\n\n\n')
        print(json.dumps(wdtd, indent=2))

        #TODO read bias


    allvalidwords_test, allvalidwordscounts_test = extractwords_for_test(test_by_class)
    group_to_test = [allvalidwords_test, allvalidwordscounts_test]

    filesign = {}
    classify(wdtd, biastd, group_to_test, 'truthful', filesign)
    classify(wdpn, biaspn, group_to_test, 'positive', filesign)

    with open('percepoutput.txt', 'w') as percepoutput_w:
        for file, value in filesign.items():
            percepoutput_w.write('{}     {}      {}\n'.format(value[0], value[1], file))

    return [all_files, test_by_class, train_by_class, allvalidwords_test, allvalidwordscounts_test, allvalidwords_train,
            allvalidwordscounts_train, priorsNC, sizeofall, test_by_class_for_analysis]


def extractstopwords():
    with open('./stopwords.txt') as stopwordsfile:
        stopwords = stopwordsfile.readline()
    return stopwords


def extractwords(group_to_test):
    stopwords = extractstopwords()
    allvalidwords = []
    allvalidwordscounts = {'positive_polaritydeceptive_from_MTurk': {},
                           'positive_polaritytruthful_from_TripAdvisor': {},
                           'negative_polaritydeceptive_from_MTurk': {},
                           'negative_polaritytruthful_from_Web': {}}

    for key, value in group_to_test.items():
        for f in value:
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
                            except:
                                allvalidwordscounts[key][token] = 1
    with open('nbmodel.txt', 'w') as nbmodel_write:
        for term in allvalidwords:
            nbmodel_write.write(term + '\n')
        nbmodel_write.write('\n\nend of allvalidwords***************\n')
        nbmodel_write.write(json.dumps(allvalidwordscounts, indent=2))
    return [allvalidwords, allvalidwordscounts]

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

    return [allvalidwords, allvalidwordscounts]

def classify(wd, bias, group_to_test, phase, filesign):
    allvalidwords_test, allvalidwordscounts_test = group_to_test

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

    with open('nboutput.txt', 'w') as nbmodel_write:
        for file, val in outputtags.items():
            nbmodel_write.write('{}     {}      {}\n'.format(val[0], val[1], file))
    return outputtags

def analysis(outputtags, test_by_class_for_analysis):
    tps = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    fps = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    fns = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    precision = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    recall = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
    f1 = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}

    for file, value in outputtags.items():
        # print(test_by_class_for_analysis[file])
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
        precision[iterator] = tps[iterator] / (tps[iterator] + fps[iterator])
    # print(tps)
    # print(fps)
    # print(fns)
    for iterator in recall.keys():
        recall[iterator] = tps[iterator] / (tps[iterator] + fns[iterator])

    for iterator in f1.keys():
        f1[iterator] = (2 * (precision[iterator] * recall[iterator])) / (precision[iterator] + recall[iterator])
    # print(precision)
    # print(recall)
    # print(f1)
    pass


if __name__ == "__main__":
    all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
    modeltoread = sys.argv[1]

    all_files, test_by_class, train_by_class, allvalidwords_test, \
    allvalidwordscounts_test, allvalidwords_train, allvalidwordscounts_train, priorsNC, sizeofall, test_by_class_for_analysis = readfiles(all_files, modeltoread)
    # outputtags = compute_multinomial(allvalidwords_test, allvalidwordscounts_test, allvalidwords_train, allvalidwordscounts_train, priorsNC, sizeofall)
    # analysis(outputtags, test_by_class_for_analysis)