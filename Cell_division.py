import numpy as np
#import maplotlib.pyplot as pyplot
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import math

def getTimeDiff(date1, date2):
    date1 = date1 / 1000
    date2 = date2 / 1000
    date1=datetime.datetime.fromtimestamp(date1)
    date2=datetime.datetime.fromtimestamp(date2)
    delta=date2-date1
    return divmod(delta.total_seconds(),60)[1]

# inputFile inputFile
# num_tf num of time frame
# num_cell num of cell to divide
# axis axis to analyze

# assume taking x = [0, 2], y = [-2, 2], z = [-2, 2]
# assume 4 blocks

def dataTransform(inputFile, inProximity, num_tf = 30, num_cell = 10, dimensionality = '3d'):
    data_row = []
    inFile = open(inputFile, 'r')
    lines = inFile.readlines()
    lines = [line.strip().split(' ') for line in lines]

    x, y, z = [], [], []
    currTime = None
    timeCount = 0
    for line in lines:
        line = [float(element.split('::')[1]) for element in line]
        if currTime is None:
            currTime = line[-1]
        elif getTimeDiff(int(currTime), int(line[-1])) > inProximity:
            count = np.zeros(shape = (num_cell ** 3, ), dtype = int)

            for x1, y1, z1 in zip(x, y, z):
                index = math.floor(x1 / (2 / num_cell)) * (num_cell ** 2) + math.floor((2 + y1) / (4 / num_cell)) + math.floor((2 - z1) / (4 / num_cell)) * num_cell
                count[index] += 1

            total = sum(count)
            count = [float(i)/total for i in count]

            #print (len(x))
            data_row.extend(count)

            timeCount += 1
            currTime = line[-1]
            x, y, z = [], [], []
        else:
            if line[0] >= 0 and line[0] < 2 and line[1] >= -2 and line[1] < 2 and line[2] >= -1 and line[2] < 1:
                x.append(line[0])
                y.append(line[1])
                z.append(line[2])

        if timeCount >= num_tf:
            break


    #print (len(data_row))
    return data_row


def main():
    trainset, testset = [], []
    num_ges = 50
    gesture_ls = ['alarm', 'rain', 'push', 'today', 'year']
    ges_ls_2 = ['thankyou', 'hello', 'help']
    ges_ls_3 = ['today', 'shelf', 'when', 'year']
    # 0 for rain 1 for push
    for i in range(len(gesture_ls)):
        break
        for j in range(num_ges):
            if j < 9:
                num = '0' + str(j + 1)
            else:
                num = str(j + 1)
            file = './20inches/riley_' + gesture_ls[i] + '_' + num + '_exp_11_06_2019.txt'
            row = dataTransform(file, 0.1)
            if gesture_ls[i] == 'today':
                row.append(3)
            elif gesture_ls[i] == 'year':
                row.append(4)
            else:
                break
            #row.append(i)
            trainset.append(row)
            #print ('len row', len(row))

        print (gesture_ls[i], 'done')

    for i in range(len(ges_ls_2)):
        break
        for j in range(num_ges):
            if j < 9:
                num = '0' + str(j + 1)
            else:
                num = str(j + 1)
            if ges_ls_2[i] == 'help' and j == 0:
                continue
            file = './20inches/riley_' + ges_ls_2[i] + '_' + num + '_exp_12_06_2019.txt'
            row = dataTransform(file, 0.1)
            row.append(i + 5)
            trainset.append(row)

        print (ges_ls_2[i], 'done')

    for i in range(len(ges_ls_3)):
        for j in range(num_ges):
            if j < 9:
                num = '0' + str(j + 1)
            else:
                num = str(j + 1)
            file = './exp_19_06/fred_' + ges_ls_3[i] + '_' + num + '_exp.txt'
            row = dataTransform(file, 0.1)
            if ges_ls_3[i] == 'today':
                row.append(3)
            elif ges_ls_3[i] == 'year':
                row.append(4)
            else:
                break
                row.append(i + 8)
            trainset.append(row)

        print (ges_ls_3[i], 'done')
    for i in range(len(gesture_ls)):
        for j in range(num_ges):
            if j < 9:
                num = '0' + str(j + 1)
            else:
                num = str(j + 1)
            file = './20inches/riley_' + gesture_ls[i] + '_' + num + '_exp_11_06_2019.txt'
            row = dataTransform(file, 0.1)
            if gesture_ls[i] == 'today':
                row.append(3)
            elif gesture_ls[i] == 'year':
                row.append(4)
            else:
                break
            #row.append(i)
            trainset.append(row)
            #print ('len row', len(row))

        print (gesture_ls[i], 'done')

    trainset = np.array(trainset)
    print (trainset.shape)
    #X, y = trainset[:, :trainset.shape[1] - 1], trainset[:, trainset.shape[1] - 1]
    #train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.5)


    # predict fred based on riley
    ratio = 1 / 5
    testset, trainset = trainset[math.floor(trainset.shape[0] // 2) - 1:, :], trainset[:math.floor(trainset.shape[0] // 2) - 1, :]
    print (testset.shape, trainset.shape)
    for k in [3, 4]:
        args = np.argwhere(testset[:, testset.shape[1] - 1] == k)
        args = args[:(math.floor(len(args) * ratio) - 1)]
        print ([i[0] for i in np.take(testset, args, axis = 0)])
        trainset = np.append(trainset, [i[0] for i in np.take(testset, args, axis = 0)], axis = 0)
        testset = np.delete(testset, args, axis = 0)
    train_X, train_y = trainset[:, :trainset.shape[1] - 1], trainset[:, trainset.shape[1] - 1]
    test_X, test_y = testset[:, :testset.shape[1] - 1], testset[:, testset.shape[1] - 1]
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    #print (X.shape, y.shape)
    print ('train_y', Counter(train_y))
    print ('test_y', Counter(test_y))

    #train_X, train_y = trainset[:, :trainset.shape[1] - 1], trainset[:, trainset.shape[1] - 1]
    #test_X, test_y = testset[:, :trainset.shape[1] - 1], testset[:, trainset.shape[1] - 1]

    #print (train_X, train_y)
    rf = RandomForestClassifier(n_estimators = 1000)
    rf.fit(train_X, train_y)
    res = rf.predict(test_X)
    print (res)
    print (Counter(res))
    accuracy = accuracy_score(res, test_y)
    print ('rf accuracy', accuracy)
    for k in range(5):
        knn = KNeighborsClassifier(n_neighbors = k + 1)
        knn.fit(train_X, train_y)
        res = knn.predict(test_X)
        print(res)
        accuracy = accuracy_score(res, test_y)
        print ('knn', k + 1, accuracy)

if __name__ == '__main__':
    main()
