# for types in typeofclasses:
#     for classlist in classlists:
#         for type in types:
#             if type in classlist:
#                 for file, val in allvalidwordscounts_test[classlist].items():
#                     multinomresult[file] = {'positive': 0, 'negative': 0, 'truthful': 0, 'deceptive': 0}
#                     for word, count in file[val]:  # not done yet
#                         multinomresult[file][type] += (
#                                 math.log(allvalidwordscounts_train[classlist][file][word] + 1) / allwordsnum[
#                             classlist] + sizeofdict * val)
#                 multinomresult += math.log(priorsNC[key] / sizeofall)

# use this file to learn naive-bayes classifier
# Expected: generate nbmodel.txt

# import sys


# if __name__ == "main":
#    model_file = "nbmodel.txt"
#    input_path = str(sys.argv[0])

# withnum = re.compile(r'\d+')
# print('******************************')
#             print(ylabel)
#             print(file)
#             print('******************************')

# with open('wdcount.txt', 'w') as wdcountfile:
    #     wdcountfile.write(json.dumps(wd, indent=2))

# print('\n\n\n\ndudul')
    # print(json.dumps(wd, indent=2))
    # print('dudul')
    # with open('wdcount.txt', 'w') as wdcountfile:
    #     wdcountfile.write(json.dumps(wd, indent=2))


# print('allvalidwordscountsdocwc')
    # print(json.dumps(allvalidwordscountsdocwc, indent=2))
    # with open('allvalidwordscountsdocwc.txt', 'w') as docwcfile:
    #     docwcfile.write(json.dumps(allvalidwordscountsdocwc, indent=2))
    # print('allvalidwordscountsdocwc')
    #
    # with open('nbmodel.txt', 'w') as nbmodel_write:
    #     for term in allvalidwords:
    #         nbmodel_write.write(term + '\n')
    #     nbmodel_write.write('\n\nend of allvalidwords***************\n')
    #     nbmodel_write.write('sizeofall: {}\n'.format(sizeofall))
    #     nbmodel_write.write('\n\nstart of word counts***************\n\n')
    #     nbmodel_write.write(json.dumps(allvalidwordscounts, indent=2))