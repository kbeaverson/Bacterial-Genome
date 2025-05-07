def encode_seq(seq):
    label_enc = {'A':1, 'C':2, 'G':3, 'T':4}
    return [label_enc.get(x.upper(), 5) for x in seq]