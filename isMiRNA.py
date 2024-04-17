import getopt
import pickle
import re
import subprocess
import sys

import joblib
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import TensorDataset, DataLoader

import generateFeatures
from model import TransformerCNN
from genFeatures.parameters import ParameterParser
from precessor import seq2num, str2num, convloops, to_categorical


def pads(sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0, ):
 if maxlen is None:
  maxlen = max(len(seq) for seq in sequences)
 padded_sequences = np.zeros((len(sequences), maxlen), dtype=dtype)

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

  padded_sequences[i, -len(padded_seq):] = padded_seq

 return padded_sequences
def one_hot(X_enc, DIM_ENC):
 X_enc_len = len(X_enc)
 X_enc_vec = np.zeros((X_enc_len, DIM_ENC))
 X_enc_vec[np.arange(np.nonzero(X_enc)[0][0], X_enc_len), np.int32(
  [X_enc[k] - 1 for k in np.nonzero(X_enc)[0].tolist()])] = 1

 return X_enc_vec.tolist()

def one_hot_wrap(X_encs, MAX_LEN, DIM_ENC):
 num_X_encs = len(X_encs)
 X_encs_padded = pads(X_encs, maxlen=MAX_LEN, dtype='int8')
 X_encs_ = np.zeros(num_X_encs).tolist()
 for i in range(num_X_encs):
  X_encs_[i] = one_hot(X_encs_padded[i], DIM_ENC)

 return np.int32(X_encs_)

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
def seq2str(seqs):
 secondary_structures = []
 mfes = []

 for a_seq in seqs:
  result = subprocess.run(["RNAfold"], input=a_seq, capture_output=True, text=True)

  lines = result.stdout.split('\n')
  line = lines[1]
  structure = line.split(' ')[0]
  secondary_structures.append(structure)
  pattern = r"\(\s*([-+]?\d+\.\d+)\)"
  matches = re.findall(pattern, line)

  if matches:
   mfe = float(matches[0])
  mfes.append(mfe / len(a_seq))
 return secondary_structures, mfes
def import_seq(filename):
 seqs = []
 for record in SeqIO.parse(filename, "fasta"):
  a_seq = str(record.seq)
  seqs.append(a_seq)
 Y_test = to_categorical([0] * len(seqs), num_classes=2)
 return seqs, Y_test



def saveBestK(K):
 F = open('selectedIndexForEvaluation.txt', 'w')
 ensure = True
 for i in K:
  if ensure:
   F.write(str(i))
  else:
   F.write(',' + str(i))
  ensure = False
 F.close()
def selectKImportance(model, X):
 importantFeatures = model.feature_importances_
 Values = np.sort(importantFeatures)[::-1]  # SORTED
 K = importantFeatures.argsort()[::-1][:len(Values[Values > 0.00])]

 return X[:, K]
def genFeature(X, Y):
 args = ParameterParser()
 T = generateFeatures.gF(args, X, Y)
 X_train = T[:, :-1]
 if np.any(X_train):
  pass
 else:
  print("x_train is empty, exiting...")
  return X_train
 Y_train = T[:, -1]
 print('Features extraction ends.')
 print('[Total extracted feature: {}]\n'.format(X_train.shape[1]))
 with open(selectedModel, 'rb') as File:
  featureSelectedModel = joblib.load(File)
 X = selectKImportance(featureSelectedModel, X_train)
 print('Features selection ends.')
 print('[Total selected feature: {}]\n'.format(X.shape[1]))
 print('Converting (optimum) CSV is begin.')
 bn_size = X.shape[1] + 1
 return X, bn_size


def import_data(pos_file, MAX_LEN, DIM_ENC):
 seqs, Y_train = import_seq(pos_file)

 y_binary = np.argmax(Y_train, axis=1)
 y = 1 - y_binary
 features, bn_size = genFeature(seqs, y)
 strs, mfes = seq2str(seqs)
 mfe_dataset = np.array(mfes)
 mfe_2d = mfe_dataset[:, np.newaxis]

 feature_dataset = np.concatenate((mfe_2d, features), axis=1)
 X_dataset = encode(seqs, strs)
 X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
 return X_train, Y_train, feature_dataset, bn_size

def predict_results(x_dataset, f_dataset, model_path, bn_size):
 try:
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("using device:", device)
  model = TransformerCNN(bn_size, device)
  model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
  model.eval()
 except Exception:
  print("The model file ", model_path, " doesn't exist!")
  exit(1)
 x_tensor = torch.Tensor(x_dataset)
 f_tensor = torch.Tensor(f_dataset)
 dataset = TensorDataset(x_tensor, f_tensor)
 test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

 predictions = []
 preall = np.empty((0, 2))
 with torch.no_grad():
  for x, f in test_loader:
   outputs = model(x, f)
   _, predicted = torch.max(outputs, 1)
   predictions.extend(outputs.argmax(1).tolist())
   z_np = outputs.detach().cpu().numpy()
   preall = np.append(preall, z_np, axis=0)
 return preall

def seq_process(arg):
 new_filename = "new_sequence.fa"
 with open(new_filename, 'w') as outfile:
  outfile.write(">Sequence\n")
  outfile.write(arg)
 return new_filename


MAX_LEN = 400
DIM_ENC = 16

def main(argv):
 INFILE = './examples.fa'
 OUTFILE = './results.txt'
 flag = ''
 model_path = 'model/human/model_cv_human_4.pkl'

 global selectedModel
 selectedModel = 'feature_data/featureSelectedModel_cv_human_4.pkl'
 try:
  opts, args = getopt.getopt(argv, "hs:i:o:m:f:", ["sequence=", "infile=", "outfile=","model=","feature="])
  # print(opts)
 except getopt.GetoptError:
  print("isMiRNA.py -i <inputfile> -o <outfile>")
  sys.exit(2)
 for opt, arg in opts:
  # print(opts)
  if opt == '-h':
   print("isMiRNA.py -i <inputfile> -o <outfile>")
   sys.exit()
  elif opt in ("-s", "--sequence"):
   INFILE = seq_process(arg)
   flag = 'seq'
   seq = arg
   # print(arg)
  elif opt in ("-i", "--ifile"):
   INFILE = arg
   # print(arg)
  elif opt in ("-o", "--ofile"):
   OUTFILE = arg
   # print(arg)
  elif opt in ("-m", "--model"):
   model_path = arg
   # print(arg)
  elif opt in ("-f", "--feature"):
   selectedModel = arg
   # print(arg)

 X_TEST,_,feature_dataset, bn_size = import_data(INFILE, MAX_LEN, DIM_ENC)
 predictions = predict_results(X_TEST,feature_dataset, model_path, bn_size)
 result = np.uint8(np.argmax(predictions, axis=1))
 output_labels = np.where(result == 0, 'True', 'False')
 if OUTFILE == '':
  OUTFILE = 'results.txt'
 np.savetxt(OUTFILE, output_labels, fmt='%s')
 print("Output results in "+OUTFILE)
 if(flag=='seq'):
  print("Your input pre-miRNA sequence: {} ".format(seq))
  if result == 0:
   print("Yes,it is a pre-miRNA")
  else:
   print("No,it is not a pre-miRNA")
  exit(0)

if __name__ == "__main__":
 main(sys.argv[1:])

