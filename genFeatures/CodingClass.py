"""
Modified from source code of LGC 1.0,
downloaded from https://bigd.big.ac.cn/lgc/calculator
article: https://academic.oup.com/bioinformatics/article-abstract/35/17/2949/5288512?redirectedFrom=fulltext
"""


import operator
import argparse
import time
import numpy as np
import scipy.stats
from Bio import SeqIO


def des_start_code(codes):
	if codes in ('ATG', 'AUG'):
		return True
	return False


def des_end_code(codes):
	if codes in ('TAA', 'UAA', 'TAG', 'UAG', 'TGA', 'UGA'):
		return True
	return False


def read_by_three(string, offset):
	flag = True
	length = len(string)
	start = end = -1
	i = 0
	result = set()
	while i < length-2:
		codes = string[i:i+3]
		if des_start_code(codes) and flag:
			start = i
			flag = False
		if des_end_code(codes) and not flag:
			end = i + 2
			flag = True
		if (end > start) and (start != -1):
			result.add((start + offset, end + offset))
		i = i + 3
	return result


def get_gc(string):
	gc = ((string.count('G') + string.count('C')) / len(string)) * 100
	return gc


def get_info(string, pos):
	length = pos[1] - pos[0] + 1
	gc = get_gc(string[pos[0]:pos[1]+1])
	return str(pos[0]), str(pos[1]), str(length), str(gc)


def orf(seq):
	result_info = []
	strings = [seq, seq[1:], seq[2:]]
	for index, string in enumerate(strings):
		# print(index)
		# print(string)
		positions = read_by_three(string, index)
		positions = sorted(positions, key=operator.itemgetter(0))
		# print(positions)
		for pos in positions:
			result_info.append(get_info(seq, pos))
	# print(result_info)
	# print(len(result_info))
	return result_info


def run(seq_record):
	seq = seq_record.upper()
	measures = orf(seq)
	t = []
	if len(measures) > 0:
		length_orf = []
		gc_mea = []
		for values in measures:
			length_orf.append(int(values[2]))
			gc_mea.append(float(values[3]))
		t.append(max(length_orf))
		t.append(min(length_orf))
		t.append(np.std(length_orf))
		t.append(np.mean(length_orf))
		t.append(scipy.stats.variation(length_orf))
		t.append(max(gc_mea))
		t.append(min(gc_mea))
		t.append(np.std(gc_mea))
		t.append(np.mean(gc_mea))
		t.append(scipy.stats.variation(gc_mea))
	else:
		for _ in range(10):
			t.append(0)
	return t


######################################################
######################################################
if __name__ == '__main__':
	print("\n")
	print("###################################################################################")
	print("######################## Feature Extraction: ORF feature   ########################")
	print("##########   Arguments: python3.5 -i input -o output -l label    ##################")
	print("###################################################################################")
	print("\n")
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='Fasta format file, E.g., test.fasta')
	parser.add_argument('-o', '--output', help='CSV format file, E.g., test.csv')
	parser.add_argument('-l', '--label', help='Dataset Label, E.g., lncRNA, mRNA, sncRNA ...')
	args = parser.parse_args()
	finput = str(args.input)
	foutput = str(args.output)
	label_dataset = str(args.label)
	start_time = time.time()
	run(finput, foutput, label_dataset)
	print('Computation time %s senconds' % (time.time() - start_time))
######################################################
######################################################
