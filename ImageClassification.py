"""

Project 2 - image KNN

Find images online from a public domain database, 
then iterate through pics, get average RGB values for all

Classify pics using attributes: [average R-val, average G-val, average B-val]
                and class outcome: [0=winter, 1=fall/autumn]

First classify photos using personally-built KNN Algorithm,
then classify photos using python library KNN Algorithm,
then classify using Naive Bayes Algorithm,
then classify using SVM Algorithm

Use fisher-yates alg to produce random folds in linear time
"""
import random
from PIL import Image, ImageStat
import glob
import os
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
import warnings

#This supresses warnings, I got it online 
warnings.filterwarnings("ignore")

#Calculates the manhattan distance between two points given point attributes
def calcManhattan(dat1, dat2):
    sum = 0.0
    for x in range(0, len(dat1[0])):
        sum += abs(dat1[0][x] - dat2[0][x])
    return sum

#Takes in and randomizes a list using the Fisher-Yates shuffle algorithm
def fisher_Yates_Random_Alg(args):
    random.seed()
    count = 0
    for i in range(len(args) - 1, -1, -1):
        randNum = random.randrange(0, len(args) - count)
        args[i], args[randNum] = args[randNum], args[i]

#Splits a list into 5 folds. Extra values are appended to fifth fold
def crossValidation(args):
    split = int(len(args) / 5)
    if (split * 5 < len(args)):
        extra = len(args) - split * 5
    else: extra = 0
    newTupledArr = [[]] * 5
    fisher_Yates_Random_Alg(args)
    for i in range(0, 5):
        if i == 4:
            foldTuple = args[split * i:split * (i+1) + extra]
        else:
            foldTuple = args[split * i:split * (i+1)]
        newTupledArr[i] = foldTuple
    return newTupledArr

#Takes in two lists and populates them with file names of photos
def getAllPhotos(photoList1, photoList2):
    dirPath = input("Name of the directory containing \"Fall\" and \"Winter\" directories? -> ")
    os.chdir(dirPath)
    for x in glob.glob('**/*Fall/*.jpg'):
        photoList1.append(''.join([dirPath, "\\", x]))
    for x in glob.glob('**/*Winter/*.jpg'):
        photoList2.append(''.join([dirPath, "\\", x]))

#Method returns average RGB values of a picture given the
#picture's file location using Image and ImaageStat from PIL
def getAvgRGB(picture):
    avg = [0.0, 0.0, 0.0]
    img = Image.open(picture)
    img.load()
    avg = ImageStat.Stat._getmean(ImageStat.Stat(img))
    return avg

#Helper method for KNN alg. Takes in kClosest data points and returns
#outcome dictated by kClosest points
def kNN_Helper(closest):
    a, b = 0, 0
    for x in closest:
        result = x[1]
        if (result == 0): a += 1
        elif (result == 1): b += 1
        else: 
            print("unexpected class/outcome")
    if (a > b): return 0
    elif (a <= b): return 1

#KNN Algorithm. Takes in one point, a k-value, and a training set of points.
#Calculates the manhattan distance between the testObj and all objects in traSet.
#Then passes the kClosest points to the helper function, which returns the classified
#outcome for tstObj.
def kNN_Alg(tstObj, kVal, traSet):
    if kVal >= 1:
        kClosest = [] * kVal
    else:
        raise Exception("Must use a positive k value")
    for traObj in traSet:
        dist_current = calcManhattan(tstObj, traObj)
        if len(kClosest) < kVal:
            kClosest.append([dist_current, traObj[1]])
            kClosest.sort(key=operator.itemgetter(0), reverse=True)
        else:
            for k in range(0, len(kClosest)):
                if kClosest[k][0] > dist_current:
                    kClosest.pop(k)
                    kClosest.append([dist_current, traObj[1]])
                    kClosest.sort(key=operator.itemgetter(0), reverse=True)
                    break
    outcome = kNN_Helper(kClosest)
    return outcome

#Gets photos from a default location (NEEDS TO BE UPDATED AFTER ADDING EVERYTHING ELSE)
def getPhotos():
    fallPhotos = []
    winterPhotos = []
    getAllPhotos(fallPhotos, winterPhotos)
    avgRGBList = [] * (len(fallPhotos) + len(winterPhotos))
    print("Extracting average RGB values for all photos. Takes up to 40 seconds.\n")
    for x in fallPhotos:
        avgRGBList.append([getAvgRGB(x), 0])
    for x in winterPhotos:
        avgRGBList.append([getAvgRGB(x), 1])
    return avgRGBList

#Inside helper method, runs KNN with a single K Value
def singleKforKNN(avgRGBList, userIn, prnt=0):
    expectedOutcomes = [-1] * len(avgRGBList[0])
    traSet = avgRGBList[1] + avgRGBList[2] + avgRGBList[3] + avgRGBList[4]
    count = 0
    for x in avgRGBList[0]:
        if count == len(avgRGBList[0]): break
        expectedOutcomes[count] = kNN_Alg(x, userIn, traSet)
        count += 1
    count, total = 0, 0
    for x in range(0, len(expectedOutcomes)):
        if (expectedOutcomes[x] == avgRGBList[0][x][1]): 
            count += 1
        total += 1
    accuracy = float(count/total)
    if prnt != 0: 
        print("\nAccuracy for k == " + str(userIn) + " is: " + str(count) + "/" + str(total) + " = " + str(accuracy) + "\n")
    return (accuracy)

#Uses the KNN algorithm with K-Vals 1-10, and does so for 10 distinct
#cross folds of the dataset. Prints the average accuracy for K-Vals 1-10
#across all 10 different cross-validation sets.
def runDataReport(avgRGBList, prnt=1):
    startTime = time.process_time()
    maxAccuracy = 0.0
    dataArray = [0.0] * 100
    for x in range(0, 10):
        for k in range(1, 11):
            dataArray[x * 10 + k-1] = singleKforKNN(avgRGBList, k)
        avgRGBList = avgRGBList[0] + avgRGBList[1] + avgRGBList[2] + avgRGBList[3] + avgRGBList[4]
        avgRGBList = crossValidation(avgRGBList)
    for x in range(0, 10):
        sum = 0.0
        for y in range(0, 10):
            sum += dataArray[10 * y + x]
        if prnt==1: print("Average accuracy for k == " + str(x+1) + ": " + str(sum/10))
        if sum/10 > maxAccuracy: maxAccuracy = sum/10
    endTime = time.process_time()
    if prnt==1: print("Run-time: " + str(endTime - startTime) + " fractal seconds.    *NOTE: Module time used, NOT Module timeit")
    print("\n")
    return [3, maxAccuracy, endTime - startTime]

#Implements all algorithms from scikit learn
def useLibraryClassifier(avgRGBList, classifierVal, prnt=1):
    dataArr = []
    if classifierVal == 0:
        classifier = KNeighborsClassifier()
    elif classifierVal == 1:
        classifier = GaussianNB()
    elif classifierVal == 2:
        classifier = SVC()
    startTime = time.process_time()
    for currentFold in range(0, 10):
        x_train, y_train, x_test, y_test = [], [], [], []
        for y in range(1, 5):
            for x in range(0, len(avgRGBList[y])):
                x_train.append(avgRGBList[y][x][0])
                y_train.append(avgRGBList[y][x][1])
        for z in range(0, len(avgRGBList[0])):
            x_test.append(avgRGBList[0][z][0])
            y_test.append(avgRGBList[0][z][1])
        classifier.fit(x_train, y_train)
        if classifierVal == 0:
            for k in range(0, 10):
                classifier.n_neighbors = k + 1
                accuracy = float(classifier.score(x_test, y_test))
                dataArr.append(accuracy)
        else:
            accuracy = float(classifier.score(x_test, y_test))
            dataArr.append(accuracy)
        avgRGBList = avgRGBList[0] + avgRGBList[1] + avgRGBList[2] + avgRGBList[3] + avgRGBList[4]
        avgRGBList = crossValidation(avgRGBList)
    for x in range(0, 10):
        sum = 0.0
        max = 0.0
        avgAccuracy = 0.0
        if classifierVal == 0:
            for y in range(0, 10):
                sum += dataArr[10 * y + x]
            if (sum/10 > max):
                max = sum/10
                avgAccuracy = max
            if prnt==1: print("Average library-KNN accuracy for k == " + str(x+1) + ": " + str(sum/10))
        elif classifierVal == 1:
            if prnt==1: print("Average Gaussian Naive Beyers accuracy for fold " + str(x+1) + ": " + str(dataArr[x]))
            for x in dataArr:
                sum += x
            avgAccuracy = sum/10
        elif classifierVal == 2:
            if prnt==1: print("Average Support Vector Classifier accuracy for fold " + str(x+1) + ": " + str(dataArr[x]))
            for x in dataArr:
                sum += x
            avgAccuracy = sum/10
    endTime = time.process_time()
    if prnt==1: print("Run-time: " + str(endTime - startTime) + " fractal seconds.    *NOTE: Module time used, NOT Module timeit")    
    print("\n")
    return [classifierVal, avgAccuracy, endTime - startTime]

#Tests each classifier algorithm over 10 different sets. Then outputs a graph,
#showing the average accuracy of that classifier over 10 sets and it's overall running time.
def graphAll(avgRGBList):
    data = np.zeros((4, 3))
    n_groups = 4
    for x in range(0, 3):
        data[x] = useLibraryClassifier(avgRGBList, x, prnt=0)
    data[3] = runDataReport(avgRGBList, prnt=0)

    accuracies = data[:, 1]
    runTimes = data[:, 2]

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, accuracies, bar_width, alpha=opacity, color='b', label='Average Accuracies')
    rects2 = plt.bar(index + bar_width, runTimes, bar_width, alpha=opacity, color = 'g', label='Runtimes (fractal sec)')

    plt.xlabel('Classifier Algorithm')
    plt.ylabel('Accuracy/Fractal seconds')
    plt.title('Accuracy and Runtime for 4 Classifiers')
    plt.xticks(index + bar_width, ('PyLibKNN', 'PyLibNaive-Beyers', 'PyLibSVM', 'Personal KNN'))
    plt.legend()

    plt.tight_layout()
    plt.show()

avgRGBList = getPhotos()
userIn = 0
avgRGBList = crossValidation(avgRGBList)

while (1):
    userIn = int(input("What value of k for kNN Alg? *inputs:\n(-1): quit\n(-2): re-shuffle cross validation sets\n" + 
                       "(-3): Run 5-fold cross-validated data report using k-vals 1 thru 10 with personal KNN classifier\n" + 
                       "(-4): Run 5-fold cross-validated data report using k-vals 1 thru 10 with py library KNN classifier\n" +
                       "(-5): Run 5-fold cross-validated data report using Gaussian Naive Bayes classifier\n" +
                       "(-6): Run 5-fold cross-validated data report using Support Vector Machine classifier\n" +
                       "(-7): Run data report using all classifiers, also outputs a graph showing run-time and accuracy\n" +
                       "(Any positive value): Run personal KNN classifier with k == (any positive value)\n\n"))
    if userIn == -1: break
    elif userIn == -2: 
        avgRGBList = avgRGBList[0] + avgRGBList[1] + avgRGBList[2] + avgRGBList[3] + avgRGBList[4]
        avgRGBList = crossValidation(avgRGBList)
        print("Re-shuffled using Fisher-Yates shuffle.\n")
    elif userIn == -3:
        runDataReport(avgRGBList)
    elif userIn == -4:
        useLibraryClassifier(avgRGBList, 0)
    elif userIn == -5:
        useLibraryClassifier(avgRGBList, 1)
    elif userIn == -6:
        useLibraryClassifier(avgRGBList, 2)
    elif userIn == -7:
        graphAll(avgRGBList)
    else:
        singleKforKNN(avgRGBList, userIn, prnt=1)


        
    
