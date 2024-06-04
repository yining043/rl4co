# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:45:10 2020

@author: Kenneth
"""

"""
Import modules needed.
"""
import itertools
import numpy as np
from tqdm import tqdm

"""
This function reads a FASTQ file.
"""
def readFastq(filename):
    sequences = []
    qualities = []
    with open(filename) as fh:
        while True:
            fh.readline()  # skip name line
            seq = fh.readline().rstrip()  # read base sequence
            fh.readline()  # skip placeholder line
            qual = fh.readline().rstrip() # base quality line
            if len(seq) == 0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities

"""
This function finds the length of the longest suffix of a which overlaps with a prefix of b.
"""
def overlap(a, b, min_length=3):
    """ Return length of longest suffix of 'a' matching
        a prefix of 'b' that is at least 'min_length'
        characters long.  If no such overlap exists,
        return 0. """
    start = 0  # start all the way at the left
    while True:
        start = a.find(b[:min_length], start)  # look for b's suffx in a
        if start == -1:  # no more occurrences to right
            return 0
        # found occurrence; check for full suffix/prefix match
        if b.startswith(a[start:]):
            return len(a)-start
        start += 1  # move just past previous match

"""
This function finds the set of shortest common superstrings of given strings.
Note that the given strings must have the same length.
"""
def scs(ss):
    """ Returns shortest common superstring of given
        strings, which must be the same length """
    shortest_sup = []
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss)-1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i+1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i+1][olen:]
        if len(shortest_sup) == 0 or len(sup) < len(shortest_sup[0]):
            shortest_sup = [sup]  # found shorter superstring
        elif len(sup) == len(shortest_sup[0]):
            shortest_sup.append(sup)
    return shortest_sup  # return shortest

"""
Given a set of reads, this function finds the pair which overlap the most and calculates the length of the overlap.
"""
def pick_maximal_overlap(reads, k):
    reada, readb = None, None
    best_olen = 0
    for a,b in itertools.permutations(reads, 2):
        olen = overlap(a, b, k)
        if olen > best_olen:
            reada, readb = a, b
            best_olen = olen
    return reada, readb, best_olen

"""
This function implements the greedy shortest common superstring algorithm.
"""
def greedy_scs(reads, k):
    read_a, read_b, olen = pick_maximal_overlap(reads, k)
    while olen > 0:
        # print(len(reads))
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        read_a, read_b, olen = pick_maximal_overlap(reads, k)
    return ''.join(reads)

"""
This is an accelerated version of pick_maximal_overlap(reads, k).
This is achieved by building an k-mer index so that not every permutation of reads is considered.
"""
def pick_maximal_overlap_index(reads, k):
    index = {}
    for read in reads:
        kmers = []
        for i in range(len(read) - k + 1):
            kmers.append(read[i:i+k])
        for kmer in kmers:
            if kmer not in index:
                index[kmer] = set()
            index[kmer].add(read)
    for read in reads:
        for i in range(len(read)-k+1):
            dummy = read[i:i+k]
            if dummy not in index:
                index[dummy] = set()
            index[dummy].add(read)
    reada, readb = None, None
    best_olen = 0
    for a in reads:
        for b in index[a[-k:]]:
            if a != b:
                olen = overlap(a, b, k)
                if olen > best_olen:
                    reada, readb = a, b
                    best_olen = olen
    return reada, readb, best_olen

"""
This function implements the greedy shortest common superstring algorithm using an accelerated version of pick_maximal_overlap(reads, k).
"""
def greedy_scs_index(reads, k):
    read_a, read_b, olen = pick_maximal_overlap_index(reads, k)
    while olen > 0:
        # print(len(reads))
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        read_a, read_b, olen = pick_maximal_overlap_index(reads, k)
    return ''.join(reads)      

def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    batch_size = x[list(x.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


codes = np.load('./data/ssp_100_15.npz')["codes"]
instance = []
for i in tqdm(range(10000)):
    ssp = codes[i].astype(int).tolist()
    ss = []
    for j in ssp:
        ss.append("".join(str(x) for x in j))
    instance.append(ss)
print(len(instance), len(instance[0]), len(instance[0][0]))

from tqdm import tqdm 
ans_greedy = []
k =20
for i in tqdm(range(10000)):
    gr_ssstr = greedy_scs(instance[i], k)
    ans_greedy.append(len(gr_ssstr))
ans_greedy = np.array(ans_greedy)

print(np.mean(ans_greedy))
np.savez(f'greedy_{k}-mers_output.npz', ans_greedy)

# 1-mers: about 1h50m for 10,000 instances, the average is 905.1268
# 2-mers: about 1h8m  for 10,000 instances, the average is 920.365
# 3-mers: about 43m02s for 10,000 instances, the average is 940.528
# 5-mers: about 20m59s for 10,000 instances, the average is 977.8808
# 10-mers: about 6m10s for 10,000 instances, the average is 1264.7653
# 14-mers: about 1m36s for 10,000 instances, the average is 1444.2949
# 15-mers: about 38s for 10,000 instances, the average is 1483.6965