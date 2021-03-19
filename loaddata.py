import numpy as np

path = 'msl-images/msl_synset_words-indexed.txt'

dat = np.genfromtxt(path, dtype=str, delimiter="  ", usecols=(0, -1))
dat = np.char.strip(dat)

