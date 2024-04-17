import re
import subprocess

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import generateFeatures
from feat import genFeature
from genFeatures import save
from genFeatures.parameters import ParameterParser
from genFeatures.save import savetestCSV


def import_seq(filename):
    seqs = []
    for record in SeqIO.parse(filename, "fasta"):
        a_seq = str(record.seq)
        seqs.append(a_seq)
    Y_test = to_categorical([0] * len(seqs), num_classes=2)
    return seqs, Y_test


def read_new_csv(positive,negative):
    try:
        hsa = pd.read_csv(positive)
        pseudo = pd.read_csv(negative)
    except IOError:
        print("Exception:hsa_new.csv or pseudo_new.csv file does not exist!")
        exit(2)
    # merge the positive and negative data into a dataset
    dataset = hsa.append(pseudo)
    # shuffle the order
    dataset = shuffle(dataset,random_state = 42)
    # 添加日志信息
    print("Dataset is prepared! Shape:", dataset.shape)
    # print("dataset is prepared!")
    return dataset

def mfe_to_vector(mfe_value, seqs):
    # 将自由能值归一化到一个范围，例如 -1 到 1
    # normalized_mfe = (mfe_value - MIN_MFE) / (MAX_MFE - MIN_MFE) * 2 - 1
    #
    # # 将归一化后的自由能值转为 PyTorch 的张量
    # mfe_tensor = torch.tensor(normalized_mfe, dtype=torch.float32)
    # mfe = float(matches[0])
    # mfes.append(mfe / len(a_seq))
    mfe_tensor = []
    for line, seq in zip(mfe_value, seqs):
        mfe_tensor.append(line / len(seq))

    return mfe_tensor

def transform_ydata(df_column):
    y_cast = {True: [1, 0], False: [0, 1]}
    y_dataset = []
    for line in df_column:
        y_dataset.append(y_cast[line])
    return y_dataset
def seq2vector(seqs):
    seq_tensor = []
    for line in seqs:
        seq_tensor.append(line)

    return seq_tensor


def import_cv_seqences(file_path):
    # dataframe = read_new_csv(file_path)
    dataframe = pd.read_csv(file_path)
    x_dataset = seq2vector(dataframe["HairpinSequence"])
    y_dataset = transform_ydata(dataframe["Classification"])
    mfes = mfe_to_vector(dataframe["MFE"], dataframe["HairpinSequence"])
    second_structs = dataframe['RNAFolds']
    # mfe = mfe_to_vector(dataframe["MFE"])
    # # transform into numpy array
    # x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)
    # mfe_dataset = np.array(mfe)
    # print(x_dataset)
    # print(y_dataset)
    print("data vectorization finished!")
    return x_dataset, mfes, second_structs, y_dataset


def import_seqences(positive_file_path, negative_file_path):
    dataframe = read_new_csv(positive_file_path, negative_file_path)
    x_dataset = seq2vector(dataframe["HairpinSequence"])
    y_dataset = transform_ydata(dataframe["Classification"])
    mfes = mfe_to_vector(dataframe["MFE"], dataframe["HairpinSequence"])
    second_structs = dataframe['RNAFolds']
    # mfe = mfe_to_vector(dataframe["MFE"])
    # # transform into numpy array
    # x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)
    # mfe_dataset = np.array(mfe)
    # print(x_dataset)
    # print(y_dataset)
    print("data vectorization finished!")
    return x_dataset,mfes,second_structs, y_dataset

def seq2str(seqs):
    secondary_structures = []
    mfes = []

    for a_seq in seqs:
        # 使用 RNAfold 工具预测二级结构
        result = subprocess.run(["RNAfold"], input=a_seq, capture_output=True, text=True)

        # 获取预测的二级结构，通常在输出中可以找到包含点括号的部分
        lines = result.stdout.split('\n')
        line = lines[1]
        structure = line.split(' ')[0]
        secondary_structures.append(structure)
        pattern = r"\(\s*([-+]?\d+\.\d+)\)"
        matches = re.findall(pattern, line)

        if matches:
            mfe = float(matches[0])
        mfes.append(mfe/len(a_seq))
    return secondary_structures,mfes


# 20231225_sam++ <<

def seq2num(a_seq):
    ints_ = [0] * len(a_seq)
    for i, c in enumerate(a_seq.lower()):
        if c == 'c':
            ints_[i] = 1
        elif c == 'g':
            ints_[i] = 2
        elif (c == 'u') | (c == 't'):
            ints_[i] = 3

    return ints_


def str2num(a_str):
    ints_ = [0] * len(a_str)
    for i, c in enumerate(a_str.lower()):
        if c == ')':
            ints_[i] = 1
        elif c == '.':
            ints_[i] = 2
        elif c == ':':
            ints_[i] = 3

    return ints_


# convert loops from '.'s to ':'s
def convloops(a_str):
    chrs_ = a_str[:]
    prog = re.compile('\(+(\.+)\)+')
    for m in prog.finditer(a_str):
        # print m.start(), m.group()
        chrs_ = "".join((chrs_[:m.regs[1][0]], ':' * (m.regs[1][1] - m.regs[1][0]), chrs_[(m.regs[1][1]):]))

    return chrs_


def encode(seqs, strs):
    if not isinstance(seqs, list):
        print("[ERROR:encode] Input type must be multidimensional list.")
        return

    if len(seqs) != len(strs):
        print("[ERROR:encode] # sequences must be equal to # structures.")
        return

    encs = []
    for a_seq, a_str in zip(seqs, strs):
        encs.append([4 * i_seq + i_str + 1 for i_seq, i_str in zip(seq2num(a_seq), str2num(convloops(a_str)))])

    return encs


def import_pos_data(filename):
    encs = []
    seqs = import_seq(filename)
    Y = to_categorical([0] * len(seqs), num_classes=2)
    y_binary = np.argmax(Y, axis=1)
    y = 1 - y_binary
    strs,mfes = seq2str(seqs)
    features = genFeature(seqs,y)
    mfe_dataset = np.array(mfes)
    # 将 b 转换为二维数组
    mfe_2d = mfe_dataset[:, np.newaxis]

    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)

    return encode(seqs, strs),feature_dataset,Y

def import_neg_data(filename):
    encs = []
    seqs = import_seq(filename)
    Y = to_categorical([1] * len(seqs), num_classes=2)
    y_binary = np.argmax(Y, axis=1)
    y = 1 - y_binary
    strs,mfes = seq2str(seqs)
    features = genFeature(seqs,y)
    mfe_dataset = np.array(mfes)
    # 将 b 转换为二维数组
    mfe_2d = mfe_dataset[:, np.newaxis]

    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    return encode(seqs, strs),feature_dataset,Y


def evaluate(testDatasetPath, optimumDatasetPath):

    ### Test Dataset ###
    D = pd.read_csv(testDatasetPath, header=None)
    X = D.iloc[:, :-1].values
    y_test = D.iloc[:, -1].values

    F = open('selectedIndex.txt', 'r')
    v = F.read().split(',')
    v = [int(i) for i in v]
    X_test = X[:, v]

    ### --- ###


    ### Hands-on scaling on train dataset  ###

    D = pd.read_csv(optimumDatasetPath, header=None)
    X_train = D.iloc[:, :-1].values
    y_train = D.iloc[:, -1].values

    storeMeanSD = []
    for i in range(X_train.shape[1]):
        eachFeature = X_train[:, i]
        MEAN = np.mean(eachFeature)
        SD = np.std(eachFeature)
        ### Stored mean and standard deviation for each feature ###
        storeMeanSD.append((MEAN, SD))

    storeMeanSD = np.array(storeMeanSD)


    scalingX_test = []
    for i in range(X_test.shape[1]):
        eachFeature = X_test[:, i]
        v = (eachFeature - storeMeanSD[i][0]) / (storeMeanSD[i][1])
        scalingX_test.append(v)

    ### --- ###

    ## Scaling X_test using X_train ###
    X_test = np.array(scalingX_test).T
    return X_test
def saveBestK(K,signal):
 F = open(f'/selectedIndexForEvaluation_{signal}.txt', 'w')
 ensure = True
 for i in K:
  if ensure:
   F.write(str(i))
  else:
   F.write(',' + str(i))
  ensure = False
 F.close()
def selectKImportance(model, X, signal):
 # 它获取模型中每个特征的重要性分数，并将其存储在 importantFeatures 数组中
 importantFeatures = model.feature_importances_
 Values = np.sort(importantFeatures)[::-1]  # SORTED
 K = importantFeatures.argsort()[::-1][:len(Values[Values > 0.00])]
 # saveBestK(K, signal)
 # 表示从数组 X 中选择所有行，但只选择列索引为 K 中所包含的列
 return X[:, K]

def genTestFeature(X, Y, signal):
 args = ParameterParser()
 if args.isExistFeatures == 0:
     T = generateFeatures.gF(args, X, Y)
     X_train = T[:, :-1]
     if np.any(X_train):
      # 如果 x_train 不为空，则执行后续操作
      pass
     else:
      # 如果 x_train 为空，则跳出
      print("x_train is empty, exiting...")
      return X_train
     Y_train = T[:, -1]
     print('Features extraction ends.')
     print('[Total extracted feature: {}]\n'.format(X_train.shape[1]))
     if args.testDataset == 1:
         print('Converting (test) CSV is begin.')
         save.saveCSV(X_train, Y_train, 'test', signal)
         print('Converting (test) CSV is end.')
     selectedModel = f'./feature_data/featureSelectedModel_{signal}.pkl'
     with open(selectedModel, 'rb') as File:
      featureSelectedModel = joblib.load(File)
     X = selectKImportance(featureSelectedModel, X_train, signal)
     savetestCSV(X, Y_train, signal)
     print('Features selection ends.')
     print('[Total selected feature: {}]\n'.format(X.shape[1]))
     print('Converting (optimum) CSV is begin.')
 else:
     data = pd.read_csv(f"./feature_data/testOptimumDataset_{signal}.csv", header=None)
     dataset_np = data.to_numpy()
     print("测试数据集的形状:", dataset_np.shape)
     X = dataset_np[:, :-1]
     labels = dataset_np[:, -1]
     print("测试数据特征的形状:", X.shape)
     print("测试标签的形状:", labels.shape)
 return X

def import_test_data(pos_file, neg_file,MAX_LEN, DIM_ENC, signal):
    seqs,mfes, strs, Y_train = import_seqences(pos_file, neg_file)

    y_binary = np.argmax(Y_train, axis=1)
    y = 1 - y_binary
    # strs, mfes = seq2str(seqs)
    mfe_dataset = np.array(mfes)
    mfe_2d = mfe_dataset[:, np.newaxis]
    flag = "test"
    features = genTestFeature(seqs, y, signal)
    # features = evaluate(testDatasetPath, optimumDatasetPath)
    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    X_dataset = encode(seqs, strs)
    X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
    return X_train, Y_train, feature_dataset

def import_data(pos_file, neg_file, MAX_LEN, DIM_ENC,signal):
    seqs,mfes, strs, Y_train = import_seqences(pos_file, neg_file)

    y_binary = np.argmax(Y_train, axis=1)
    y = 1 - y_binary
    # strs, mfes = seq2str(seqs)
    mfe_dataset = np.array(mfes)
    mfe_2d = mfe_dataset[:, np.newaxis]
    features, bn_size = genFeature(seqs, y, signal)
    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    X_dataset = encode(seqs, strs)
    X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
    return X_train, Y_train, feature_dataset, bn_size

def import_cv_val_data(file, MAX_LEN, DIM_ENC,signal):
    seqs,mfes, strs, Y_train = import_cv_seqences(file)

    y_binary = np.argmax(Y_train, axis=1)
    y = 1 - y_binary
    # strs, mfes = seq2str(seqs)
    mfe_dataset = np.array(mfes)
    mfe_2d = mfe_dataset[:, np.newaxis]
    features = genTestFeature(seqs, y,signal)
    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    X_dataset = encode(seqs, strs)
    X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
    return X_train, Y_train, feature_dataset
def import_cv_data(file, MAX_LEN, DIM_ENC,signal):
    seqs,mfes, strs, Y_train = import_cv_seqences(file)

    y_binary = np.argmax(Y_train, axis=1)
    y = 1 - y_binary
    # strs, mfes = seq2str(seqs)
    mfe_dataset = np.array(mfes)
    mfe_2d = mfe_dataset[:, np.newaxis]
    features, bn_size = genFeature(seqs, y,signal)
    # 在轴 1 上拼接 a 和 b
    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    X_dataset = encode(seqs, strs)
    X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
    return X_train, Y_train, feature_dataset, bn_size

def dataPartition(pos_file, neg_file, MAX_LEN, DIM_ENC):
    x_dataset, y_dataset, features = import_data(pos_file, neg_file, MAX_LEN, DIM_ENC)
    # sam++ >>
    split = .8
    L = int(split * len(x_dataset))
    # sam++ <<
    print("data vectorization in function train_test_partition finished!")
    print(len(x_dataset))
    # generate test and train dataset
    # 20231225_sam++ >>
    x_test_dataset = x_dataset[L:]
    y_test_dataset = y_dataset[L:]
    x_train_dataset = x_dataset[:L]
    y_train_dataset = y_dataset[:L]
    features_test_dataset = features[L:]
    features_train_dataset = features[:L]
    print("train_test_partition finished!")
    return x_train_dataset, y_train_dataset, x_test_dataset, y_test_dataset, \
        features_test_dataset, features_train_dataset

def fold5CV(train_file, val_file, MAX_LEN, DIM_ENC,signal):
    x_train_segment, y_train_segment, feature_train_segment, bn_size = \
        import_cv_data(train_file, MAX_LEN, DIM_ENC,signal)

    x_validation_segment, y_validation_segment, feature_validation_segment = \
        import_cv_val_data(val_file, MAX_LEN,DIM_ENC,signal)
    print("数据向量化完成！")
    return x_train_segment, y_train_segment, x_validation_segment, \
           y_validation_segment, feature_train_segment, feature_validation_segment, bn_size
    # 20231228_sam++ <<




def pads(sequences,maxlen=None,dtype="int32",padding="pre",truncating="pre",value=0.0,):
  if maxlen is None:
   maxlen = max(len(seq) for seq in sequences)
  # 创建填充后的数组
  padded_sequences = np.zeros((len(sequences), maxlen), dtype=dtype)

  # 填充或裁剪序列
  for i, seq in enumerate(sequences):
   if padding == "pre":
    if truncating == "pre":
     padded_seq = seq[:maxlen]
    else:
     padded_seq = seq[-maxlen:]
   else:
    if truncating == "pre":
     padded_seq = seq[-maxlen:]
    else:
     padded_seq = seq[:maxlen]

   # 将填充后的序列赋值给填充数组
   padded_sequences[i, -len(padded_seq):] = padded_seq

  return padded_sequences

def one_hot_wrap(X_encs, MAX_LEN, DIM_ENC):
    num_X_encs = len(X_encs)
    X_encs_padded = pads(X_encs, maxlen=MAX_LEN, dtype='int8')
    X_encs_ = np.zeros(num_X_encs).tolist()
    for i in range(num_X_encs):
        X_encs_[i] = one_hot(X_encs_padded[i], DIM_ENC)

    return np.int32(X_encs_)


def one_hot(X_enc, DIM_ENC):
    X_enc_len = len(X_enc)
    X_enc_vec = np.zeros((X_enc_len, DIM_ENC))
    X_enc_vec[np.arange(np.nonzero(X_enc)[0][0], X_enc_len), np.int32(
        [X_enc[k] - 1 for k in np.nonzero(X_enc)[0].tolist()])] = 1

    return X_enc_vec.tolist()


def splitCV(data, kfold):
    kf = KFold(n_splits=kfold, shuffle=True)
    train_idx = []
    test_idx = []
    for train, test in kf.split(data):
        train_idx.append(train)
        test_idx.append(test)

    return train_idx, test_idx


def perfeval(predictions, Y_test, verbose=0):
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))
    CM = metrics.confusion_matrix(R, class_label, labels=None)

    CM = np.double(CM)
    acc = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])
    se = (CM[0][0]) / (CM[0][0] + CM[0][1])
    sp = (CM[1][1]) / (CM[1][0] + CM[1][1])
    f1 = (2 * CM[0][0]) / (2 * CM[0][0] + CM[0][1] + CM[1][0])
    ppv = (CM[0][0]) / (CM[0][0] + CM[1][0])
    mcc = (CM[0][0] * CM[1][1] - CM[0][1] * CM[1][0]) / np.sqrt(
        (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]) * (CM[0][1] + CM[1][1]) * (CM[1][0] + CM[1][1]))
    gmean = np.sqrt(se * sp)
    auroc = metrics.roc_auc_score(Y_test[:, 0], predictions[:, 0])
    aupr = metrics.average_precision_score(Y_test[:, 0], predictions[:, 0], average="micro")

    if verbose == 1:
        print("SE:", "{:.3f}".format(se), "SP:", "{:.3f}".format(sp), "F-Score:", "{:.3f}".format(f1), "PPV:",
              "{:.3f}".format(ppv), "gmean:", "{:.3f}".format(gmean), "AUROC:", "{:.3f}".format(auroc), "AUPR:",
              "{:.3f}".format(aupr))

    return [se, sp, f1, ppv, gmean, auroc, aupr, CM]


def wrtrst(filehandle, rst, nfold=0, nepoch=0):
    filehandle.write(str(nfold + 1) + " " + str(nepoch + 1) + " ")
    filehandle.write("SE: %s SP: %s F-score: %s PPV: %s g-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.3f}".format(rst[0]),
                      "{:.3f}".format(rst[1]),
                      "{:.3f}".format(rst[2]),
                      "{:.3f}".format(rst[3]),
                      "{:.3f}".format(rst[4]),
                      "{:.3f}".format(rst[5]),
                      "{:.3f}".format(rst[6])))
    filehandle.flush()
    return

def to_categorical(y, num_classes=None, dtype="float32"):
    # 若y为0则，则将第0个位置置为1，即正[1, 0]，若y为1，则将第一个位置置为1，即负样本[0, 1]
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical