import argparse

def ParameterParser():

    ######################
    # Adding Arguments
    #####################

    p = argparse.ArgumentParser(description='Features Geneation Tool from DNA, RNA, and Protein Sequences')

    p.add_argument('-seq', '--sequenceType', type=str, help='DNA/RNA/PROTEIN/PROT', default='RNA')

    p.add_argument('-pos', '--positive_file', type=str, help='~/FASTA.txt',
                   default='/Users/sam/pycharm/dnnPreMiR/src/src/miRe2e2/data_sam/hsa_new.csv')
    p.add_argument('-neg', '--negative_file', type=str, help='~/FASTA.txt',
                   default='/Users/sam/pycharm/dnnPreMiR/src/src/miRe2e2/data_sam/pseudo_new.csv')

    p.add_argument('-kgap', '--kGap', type=int, help='(l,k,p)-mers', default=5)
    p.add_argument('-ktuple', '--kTuple', type=int, help='k=1 then (X), k=2 then (XX), k=3 then (XXX),', default=3)
    p.add_argument('-kmer', '--kMer', type=int, help='Generate feature: XXX_XX', default=0, choices=[0, 1])

    p.add_argument('-full', '--fullDataset', type=int, help='saved full dataset', default=1, choices=[0, 1])
    p.add_argument('-test', '--testDataset', type=int, help='saved test dataset', default=1, choices=[0, 1])
    p.add_argument('-optimum', '--optimumDataset', type=int, help='saved optimum dataset', default=1, choices=[0, 1])

    p.add_argument('-pseudo', '--pseudoKNC', type=int, help='Generate feature: X, XX, XXX, XXX', default=1,
                   choices=[0, 1])
    p.add_argument('-zcurve', '--zCurve', type=int, help='x_, y_, z_', default=1, choices=[0, 1])
    p.add_argument('-gc', '--gcContent', type=int, help='GC/ACGT', default=1, choices=[0, 1])
    p.add_argument('-skew', '--cumulativeSkew', type=int, help='GC, AT', default=1, choices=[0, 1])
    p.add_argument('-atgc', '--atgcRatio', type=int, help='atgcRatio', default=1, choices=[0, 1])
    p.add_argument('-orf', '--orf', type=int, help='Generate feature: XXX_XX', default=1, choices=[0, 1])
    p.add_argument('-fScore', '--fickettScore', type=int, help='Generate feature: XXX_XX', default=1, choices=[0, 1])
    p.add_argument('-entropy', '--entropy_equation', type=int, help='Generate feature: XXX_XX', default=1,
                   choices=[0, 1])
    p.add_argument('-binary_fourier', '--binary_fourier', type=int, help='Generate feature: XXX_XX', default=1,
                   choices=[0, 1])
    p.add_argument('-tsallis', '--tsallisEntropy', type=int, help='Generate feature: XXX_XX', default=1,
                   choices=[0, 1])
    p.add_argument('-isExists', '--isExistFeatures', type=int, help='Generate feature: XXX_XX', default=1,
                   choices=[0, 1])
    p.add_argument('-i', '--infile', type=str, help='Generate feature: XXX_XX', default='examples.fa')
    p.add_argument('-o', '--outfile', type=str, help='Generate feature: XXX_XX', default='results.txt')
    p.add_argument('-s', '--sequence', type=str, help='Generate feature: XXX_XX', default='')
    p.add_argument("-m", "--model", type=str, help='Generate feature: XXX_XX', default='')
    p.add_argument("-f", "--feature", type=str, help='Generate feature: XXX_XX', default='')

    args = p.parse_args()

    return args
