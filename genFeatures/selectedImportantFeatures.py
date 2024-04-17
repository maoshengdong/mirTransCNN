import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from genFeatures import save

def saveBestK(K,signal):
    F = open(f'selectedIndex_{signal}.txt', 'w')
    ensure = True
    for i in K:
        if ensure:
            F.write(str(i))
        else:
            F.write(','+str(i))
        ensure = False
    F.close()

def selectKImportance(model, X, signal):

    # 它获取模型中每个特征的重要性分数，并将其存储在 importantFeatures 数组中
    importantFeatures = model.feature_importances_
    # 将特征的重要性分数按降序排列，并存储在 Values 数组中。这意味着最重要的特征将位于数组的前面。
    Values = np.sort(importantFeatures)[::-1] #SORTED

    # 计算最重要的特征的索引。它首先使用 argsort() 函数对特征重要性分数进行排序，
    # 然后将结果反转，以便最重要的特征排在前面。最后，它只选择重要性大于0.00的特征的索引
    K = importantFeatures.argsort()[::-1][:len(Values[Values>0.00])]
    # saveBestK(K,signal)
    # save.saveFeatures(K)

    # print(' --- begin --- ')
    #
    # for i in K:
    #     print(i, end=', ')
    # print()
    # print(' --- end dumping webserver (425) --- ')
    #
    # C=1
    # for value, eachk in zip(Values, K):
    #     print('rank:{}, value:{}, index:({})'.format(C, value, eachk))
    #
    #      C += 1
    # print('--- end ---')

    ##############################
    # print(Values)
    # print()
    # print(Values[Values>0.00])
    ##############################
    # 表示从数组 X 中选择所有行，但只选择列索引为 K 中所包含的列
    return X[:, K]


def importantFeatures(X, Y, signal):
    model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=50, learning_rate=1.0)  #sam--
    model.fit(X, Y)
    # 保存该模型
    import joblib
    with open(f'./feature_data/featureSelectedModel_{signal}.pkl', 'wb') as File:
        joblib.dump(model, File)

    return selectKImportance(model, X, signal)



