import sys
import os

'''
alpha = 0.005; beta = 1; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.005; beta = 5; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.005; beta = 10; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.01; beta = 1; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.01; beta = 5; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.01; beta = 10; L2 = 0; L1 = 0; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )
'''
alpha = 0.001; beta = 1; L2 = 0.01; L1 = 0.01; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

alpha = 0.001; beta = 1; L2 = 0.1; L1 = 0.1; epoch = 5; holdout = 1000; train = 'train.csv'; test = 'test.csv'
cmd = "pypy FTRL.py %s %s %s %s %s %s %s %s" % (alpha, beta, L2, L1, epoch, holdout, train, test)
os.system( cmd )

