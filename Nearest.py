import numpy as np
import math
import copy
import time
def loadsparsedata(fn):                 #parse and convert to float
    fp = open(fn, "r")
    instances = open(fn).read().split("\n")
    lines = open(fn).read().split("/n")
    newlines = lines[0].split("\n")
    slines = newlines[0].split()
    maxf = len(slines) - 1
    X = np.zeros((len(instances)-1, maxf))
    Y = np.zeros(len(instances)-1)
    parselines = open(fn).read().split()
    j = 0
    d = -1
    for i in range(0,len(parselines)):
        if i % len(slines) == 0:
            #toFloat = parselines[i].replace('.', '.')
            Y[j] = float(parselines[i])
            j += 1
    k = 0
    for i in range(0, len(parselines)):
        if i % len(slines) != 0:
            #toFloat = parselines[i].replace('.', '.')
            X[d][k] = float(parselines[i])
            k += 1
            if k % maxf == 0:
                k = 0
        elif i % len(slines) == 0:
            d += 1
    return X,Y,maxf

def Leave_One_Out(X,Y,features, curr_accuracy):
    accuracy = curr_accuracy                        #in case, keep current accuracy
    euclid_d = 0
    closest_euclid_d_holder = 0                     #keep track of euclidean distance
    correct = 0
    validationOneX = np.zeros((len(Y),len(features)))
    validationOneY = np.zeros(len(Y))
    classification = 0
    total_tests = 0
    count = 0
    for i in range(0,len(Y)):                       #populate both X and Y validation sets
        validationOneY[i] = Y[i]
    for i in range(0, len(Y)):
        k=0
        for j in features:
            validationOneX[i][k] = X[i][j]
            k+=1
    for v in range(0, len(validationOneX)):             #test each distance on set X and classify
        closest_euclid_d = 1000
        for j in range(0, len(Y)):
            if v == j:
                continue
            else:
                euclid_d = 0
                l = 0
                for k in features:
                    euclid_d += (X[j][k] - validationOneX[v][l])**2
                    l+= 1
                euclid_d = math.sqrt(euclid_d)
            if euclid_d < closest_euclid_d:
                classification = Y[j]
                closest_euclid_d = euclid_d
        total_tests += 1
        if classification == validationOneY[v]:
            correct += 1
    accuracy = correct/total_tests

    return accuracy

def forward_search(X,Y,features):
    current_set_features = []
    best_set_features = []
    best_accuracy = 0
    feature_of_interest = 0
    overall_best_accuracy = 0
    best_features = 0
    print("Beginning search")
    for i in range(0,len(features)):
        best_accuracy = 0
        feature_of_interest = 0
        best = False                    #keep track of best set
        for j in features:
            accuracy = 0
            if if_empty(current_set_features,j):        #if feature exists or is empty set
                current_set_features = feature_test(current_set_features,j,i)
                accuracy = Leave_One_Out(X,Y,current_set_features,0)
                print("             Using feature(s) ",current_set_features,"accuracy is ",accuracy*100,"%")
                if accuracy > best_accuracy:            #best accuracy amongst features
                    best_accuracy = accuracy
                    feature_of_interest = j
                    if best_accuracy > overall_best_accuracy:       #bool for finding best set
                        overall_best_accuracy = best_accuracy
                        best_features = j
                        best = True
        if best_accuracy < overall_best_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        if best:
            best_set_features.append(best_features)
        current_set_features[i] = feature_of_interest
        print("Feature set ", current_set_features," was best, accuracy is ",best_accuracy*100,"%")
    print("Finished search!! The best feature subset is ", best_set_features,", which has an accuracy of ", overall_best_accuracy*100,"%")

def feature_test(currfeat,feature,index):
    tempfeat = copy.deepcopy(currfeat)          #feature test is to append new feature to test the set
    if len(tempfeat) == 0:
        tempfeat.append(0)
        return tempfeat
    if tempfeat == None:
        tempfeat.append(0)
        return tempfeat
    tempfeat.append("t")
    tempfeat[index] = feature
    tempfeat = [x for x in tempfeat if x != "t"]
    return tempfeat


def if_empty(currfeat,feature):         #ensure non empty set or feature exists
    if currfeat == None:

        return True
    if len(currfeat) == 0:
        return True
    for i in currfeat:
        if feature == i:
            return False
    return True

def back_search(X,Y,features):
    current_set_features = features
    best_set_features = features
    best_accuracy = 0
    feature_of_interest = 0
    overall_best_accuracy = 0
    best_features = 0
    one_in_set = True
    list_of_accuracy = []                           #needed for further removal if lowest accuracy isnt found
    print("Beginning search")
    for i in range(0, len(features)):
        best_accuracy = 0
        feature_of_interest = 0
        best = False
        worst_accuracy = 100
        one_in_set = check_if_empty(current_set_features)       #without this, it will test on an empty set
        min_accuracy = 0
        diff_accuracy = 100
        for j in features:
            accuracy = 0
            if isEmpty(current_set_features):
                temp_set_features = copy.deepcopy(current_set_features)
                temp_set_features.remove(j)
                accuracy = Leave_One_Out(X,Y,temp_set_features,0)
                min_accuracy = worst_accuracy - accuracy            #minimum accuracy from initial accuracy
                print("             Remove ", j, "set is now ", temp_set_features, "accuracy ", accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if best_accuracy > overall_best_accuracy:       #finding best set with accuracy
                        overall_best_accuracy = best_accuracy
                        best = True
                if min_accuracy < diff_accuracy:
                    diff_accuracy = min_accuracy
                    feature_of_interest = j
        if one_in_set:                                      #remove feature that does not change accuracy by the least
            print("Trying to remove ",feature_of_interest)
            current_set_features.remove(feature_of_interest)
            print("Removing feature ",feature_of_interest," set is now ",current_set_features)
        if best:
            best_set_features = temp_set_features
    print("BEST SET WAS: ",best_set_features," with accuracy ", overall_best_accuracy*100,"%")

def check_if_empty(currfeat):
    if len(currfeat) == 1:
        return False
    else:
        return True
def isEmpty(currfeat):
    if currfeat == None:
        return False
    if len(currfeat) == 1:
        return False
    else:
        return True

def Joseph_prune_search(X,Y,features):
    current_set_features = []
    best_set_features = []
    best_accuracy = 0
    feature_of_interest = 0
    overall_best_accuracy = 0
    best_features = 0
    print("Beginning search")
    for i in range(0, len(features)):
        best_accuracy = 0
        feature_of_interest = 0
        best = False  # keep track of best set
        zero_acc = 0
        total_test = 0
        for j in features:
            accuracy = 0
            if if_empty(current_set_features, j):  # if feature exists or is empty set
                current_set_features = feature_test(current_set_features, j, i)
                accuracy = Leave_One_Out_Opt(X, Y, current_set_features,0, overall_best_accuracy)
                total_test += 1
                if accuracy == 0:
                    zero_acc += 1
                print("             Using feature(s) ", current_set_features, "accuracy is ", accuracy * 100, "%")
                if accuracy > best_accuracy:  # best accuracy amongst features
                    best_accuracy = accuracy
                    feature_of_interest = j
                    if best_accuracy > overall_best_accuracy:  # bool for finding best set
                        overall_best_accuracy = best_accuracy
                        best_features = j
                        best = True
        if zero_acc == total_test:
            print("No better accuracy found! All features pruned...search stopping")
            break
        if best_accuracy < overall_best_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        if best:
            best_set_features.append(best_features)
        current_set_features[i] = feature_of_interest
        print("Feature set ", current_set_features, " was best, accuracy is ", best_accuracy * 100, "%")
    print("Finished search!! The best feature subset is ", best_set_features, ", which has an accuracy of ",
          overall_best_accuracy * 100, "%")

def Leave_One_Out_Opt(X, Y, features, curr_accuracy,best_accuracy):
    accuracy = curr_accuracy  # in case, keep current accuracy
    euclid_d = 0
    closest_euclid_d_holder = 0  # keep track of euclidean distance
    incorrect = 0
    correct = 0
    validationOneX = np.zeros((len(Y), len(features)))
    validationOneY = np.zeros(len(Y))
    classification = 0
    total_tests = 0
    count = 0
    for i in range(0, len(Y)):  # populate both X and Y validation sets
        validationOneY[i] = Y[i]
    for i in range(0, len(Y)):
        k = 0
        for j in features:
            validationOneX[i][k] = X[i][j]
            k += 1
    for v in range(0, len(validationOneX)):  # test each distance on set X and classify
        closest_euclid_d = 1000
        for j in range(0, len(Y)):
            if v == j:
                continue
            else:
                euclid_d = 0
                l = 0
                for k in features:
                    euclid_d += (X[j][k] - validationOneX[v][l]) ** 2
                    l += 1
                euclid_d = math.sqrt(euclid_d)
            if euclid_d < closest_euclid_d:
                classification = Y[j]
                closest_euclid_d = euclid_d
        total_tests += 1
        if classification == validationOneY[v]:     #Prune after hitting incorrect threshold
            correct += 1
        if classification != validationOneY[v]:     #Prune after hitting incorrect threshold
            incorrect += 1
        if incorrect != 0 and (incorrect/len(Y)) > (1-best_accuracy):
            return 0
    accuracy = correct / total_tests

    return accuracy

print("Welcome to Bertie Woosters Feature Selection Algorithm.")
textfile = input("Type in the name of the file to test :  ")
(X,Y,features) = loadsparsedata(textfile)
featuresList = []
for i in range(0,features):
    featuresList.append(i)

print("Type the number of the algorithm you want to run.")
print("     1) Forward Selection")
print("     2) Backward Elimination")
print("     3) Bertie's Special Algorithm")
algorithm = input()
print("This dataset has ", len(featuresList)," features (not including the class attribute), with ",len(Y)," instances.")

if algorithm == '1':
    accuracy = Leave_One_Out(X, Y, featuresList, 0)
    start = time.time()
    print("Running nearest neighbor with all ", len(featuresList),
          "features, using “leaving-one-out” evaluation, I get an accuracy of", accuracy * 100, "%")
    accuracy = 0
    forward_search(X, Y, featuresList)
    end = time.time()
    print("time elapsed: ", end - start)
elif algorithm == '2':
    accuracy = Leave_One_Out(X, Y, featuresList, 0)

    print("Running nearest neighbor with all ", len(featuresList),
          "features, using “leaving-one-out” evaluation, I get an accuracy of", accuracy * 100, "%")
    accuracy = 0
    start = time.time()
    back_search(X, Y, featuresList)
    end = time.time()
    print("time elapsed: ", end - start)
elif algorithm == '3':
    accuracy = Leave_One_Out(X, Y, featuresList, 0)

    print("Running nearest neighbor with all ", len(featuresList),
          "features, using “leaving-one-out” evaluation, I get an accuracy of", accuracy * 100, "%")
    accuracy = 0
    start = time.time()
    Joseph_prune_search(X, Y, featuresList)
    end = time.time()
    print("time elapsed: ", end - start)






