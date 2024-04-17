def saveCSV(X,Y, type,signal):
    if type == 'full':
        # F = open('fullDataset.csv', 'w')
        F = open(f'./feature_data/fullDataset_{signal}.csv', 'w')

    else:
        if type == 'test':
            F = open(f'./feature_data/testDataset_{signal}.csv', 'w')

        else:
            if type == 'optimum':
                F = open(f'./feature_data/optimumDataset_{signal}.csv', 'w')

            else: None

    for x, y in zip(X, Y):
        for each in x:
            F.write(str(each) + ',')
        F.write(str(int(y)) + '\n')
    # for x in X:
    #     for each in x:
    #         F.write(str(each) + ',')
    #     F.write('\n')
    F.close()

# sam++ >>
def savetestCSV(X,Y, signal):
    F = open(f'./feature_data/testOptimumDataset_{signal}.csv', 'w')
    for x, y in zip(X, Y):
        for each in x:
            F.write(str(each) + ',')
        F.write(str(int(y)) + '\n')
    F.close()
# sam++ <<

def saveBestK(K):
    F = open('selectedIndex.txt', 'w')
    ensure = True
    for i in K:
        if ensure:
            F.write(str(i))
        else:
            F.write(','+str(i))
        ensure = False
    F.close()

def saveFeatures(tracking):
    F = open('trackingFeaturesStructure.txt', 'w')
    ensure = True
    for i in tracking:
        if ensure:
            F.write(str(i))
        else:
            F.write(',' + str(i))
        ensure = False
    F.close()




