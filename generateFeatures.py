import itertools
import numpy as np

from genFeatures.CodingClass import run
from genFeatures.EntropyClass import entropy_equation
from genFeatures.ExtractionTechniques import nacSeq, tncSeq, dncSeq
from genFeatures.FickettScore import calculate_sequences
from genFeatures.FourierClass import binary_fourier
from genFeatures.TsallisEntropy import tsallis_entropy
from genFeatures.kmers import findKmers

DNAelements = 'ACGT'
RNAelements = 'ACGU'
proteinElements = 'ACDEFGHIKLMNPQRSTVWY'

def sequenceType(seqType):
    if seqType == 'DNA':
        elements = DNAelements
    else:
        if seqType == 'RNA':
            elements = RNAelements
        else:
            if seqType == 'PROTEIN' or seqType == 'PROT':
                elements = proteinElements
            else:
                elements = None

    return elements


trackingFeatures = []

def gF(args, X,Y):

    elements = sequenceType(args.sequenceType.upper())

    m2 = list(itertools.product(elements, repeat=2))
    m3 = list(itertools.product(elements, repeat=3))
    m4 = list(itertools.product(elements, repeat=4))
    m5 = list(itertools.product(elements, repeat=5))

    T = []  # All instance ...

    def kmers(seq, k):
        v = []
        for i in range(len(seq) - k + 1):
            v.append(seq[i:i + k])
        return v

    def kMers(x, k):
        # from kmers import findKmers
        V = findKmers(x,k)
        for v in V:
            t.append(v[1])


    def pseudoKNC(x, k):
        ### k-mer ###
        ### A, AA, AAA

        for i in range(1, k + 1, 1):
            v = list(itertools.product(elements, repeat=i))
            # seqLength = len(x) - i + 1
            for i in v:
                # print(x.count(''.join(i)), end=',')
                t.append(x.count(''.join(i)))
        ### --- ###

    def zCurve(x, seqType):
        ### Z-Curve ### total = 3

        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A'); C = x.count('C'); G = x.count('G');

            x_ = (A + G) - (C + TU)
            y_ = (A + C) - (G + TU)
            z_ = (A + TU) - (C + G)
            # print(x_, end=','); print(y_, end=','); print(z_, end=',')
            t.append(x_); t.append(y_); t.append(z_)
            ### print('{},{},{}'.format(x_, y_, z_), end=',')
            ### --- ###
            # trackingFeatures.append('x_axis'); trackingFeatures.append('y_axis'); trackingFeatures.append('z_axis')

    def gcContent(x, seqType):

        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A');
            C = x.count('C');
            G = x.count('G');

            t.append( (G + C) / (A + C + G + TU)  * 100.0 )


    def cumulativeSkew(x, seqType):

        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A');
            C = x.count('C');
            G = x.count('G');

            GCSkew = (G-C)/(G+C)
            ATSkew = (A-TU)/(A+TU)

            t.append(GCSkew)
            t.append(ATSkew)


    def atgcRatio(x, seqType):

        if seqType == 'DNA' or seqType == 'RNA':

            if seqType == 'DNA':
                TU = x.count('T')
            else:
                if seqType == 'RNA':
                    TU = x.count('U')
                else:
                    None

            A = x.count('A');
            C = x.count('C');
            G = x.count('G');

            t.append( (A+TU)/(G+C) )

    def generateORF(x):
        V = run(x)
        # for v in V.items():
        #     t.append(v)
        t.extend(V)

    def FickettScore(x):
        V = calculate_sequences(x)
        t.extend(V)

    def entropy_Equation(x, kTuple):
        V = entropy_equation(x, kTuple)
        t.extend(V)

    def binary_Fourier(x):
        V = binary_fourier(x)
        t.extend(V)

    def tsallis_Entropy(x, kTuple):
        V = tsallis_entropy(x, kTuple)
        t.extend(V)

    def generateFeatures(kGap, kTuple, x, y):

        if args.zCurve == 1:
            zCurve(x, args.sequenceType.upper())              #3

        if args.gcContent == 1:
            gcContent(x, args.sequenceType.upper())           #1

        if args.cumulativeSkew == 1:
            cumulativeSkew(x, args.sequenceType.upper())      #2

        if args.atgcRatio == 1:
            atgcRatio(x, args.sequenceType.upper())         #1

        if args.pseudoKNC == 1:
            pseudoKNC(x, kTuple)            #k=2|(16), k=3|(64), k=4|(256), k=5|(1024);

        ##############################################################

        ##############################################################
        ##############################################################
        if args.kMer == 1:
            kMers(x,kTuple)


        if args.orf == 1:
            generateORF(x)

        if args.fickettScore == 1:
            FickettScore(x)

        if args.entropy_equation == 1:
            entropy_Equation(x,kTuple)

        if args.binary_fourier == 1:
            binary_Fourier(x)

        if args.tsallisEntropy == 1:
            tsallis_Entropy(x, kTuple)

        t.append(y)
        #######################

    for x, y in zip(X, Y):
        t = []
        generateFeatures(args.kGap, args.kTuple, x, y)
        T.append(t)

    return np.array(T)


