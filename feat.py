import numpy as np
import pandas as pd



# from generateFeatures import generateFeatures
import generateFeatures
from genFeatures import save, selectedImportantFeatures
from genFeatures.parameters import ParameterParser


def genFeature(X,Y,signal):
    args = ParameterParser()
    if args.isExistFeatures == 0:
        print('Features extracting,Please wait for some time.')
        T = generateFeatures.gF(args, X, Y)
        # T = generateFeatures.gF(args, X)
        # X_train = T
        X_train = T[:, :-1]
        if np.any(X_train):
            pass
        else:
            bn_size = 1

            print("X_train is empty, exiting...")
            return X_train, bn_size
        Y_train = T[:, -1]
        print('Features extraction ends.')
        print('[Total extracted feature: {}]\n'.format(X_train.shape[1]))
        #############################################################################

        if args.fullDataset == 1:
            print('Converting (full) CSV is begin.')
            save.saveCSV(X_train, Y_train, 'full',signal)
            # save.saveCSV(X_train, 'full')
            print('Converting (full) CSV is end.')



        # #############################################################################

        if args.optimumDataset == 1:
            print('\nFeatures selection begins. Be patient! The Machine will take some time.')
            # import selectedImportantFeatures
            X_train = selectedImportantFeatures.importantFeatures(X_train, Y_train,signal)
            # X_train = selectedImportantFeatures.importantFeatures(X_train)
            print('Features selection ends.')
            print('[Total selected feature: {}]\n'.format(X_train.shape[1]))
            print('Converting (optimum) CSV is begin.')
            save.saveCSV(X_train, Y_train, 'optimum',signal)
            # save.saveCSV(X_train, 'optimum')
            print('Converting (optimum) CSV is end.')
            #############################################################################
    else:
        data = pd.read_csv(f"./feature_data/optimumDataset_{signal}.csv", header=None)
        dataset_np = data.to_numpy()

        X_train = dataset_np[:, :-1]
        labels = dataset_np[:, -1]
    bn_size = X_train.shape[1] + 1
    return X_train, bn_size
