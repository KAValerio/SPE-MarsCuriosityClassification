import numpy as np

labpath = 'msl-images/msl_synset_words-indexed.txt'

labs = np.genfromtxt(labpath, dtype=str, delimiter="  ", usecols=(0, -1))
labs = np.char.strip(labs)

