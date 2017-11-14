from contextlib import contextmanager
import datetime, time, os
import matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np
import pandas as pd
from pandas.compat import StringIO
from scipy import misc
import tqdm

HOME = os.path.expanduser('~')
EPOCH = datetime.datetime(1970, 1, 1)


def unzip(archivepath, outdir='/tmp/data/'):
    with timeit('Unzipping %s' % archivepath):
        os.system('unzip -nq "%s" -d "%s"' % (archivepath, outdir))
    return outdir


def getDataDir(zipOutDir):
    dirs = os.listdir(zipOutDir)
    if 'IMG' not in dirs:
        assert len(dirs) > 0
        zipOutDir = os.path.join(zipOutDir, dirs[0])
    return zipOutDir


class Data(object):

    def __init__(self, *args, **kwargs):

        self.sidecamAdjustment = .05

        kwargs.setdefault('verbose', True)

        self.verbose = kwargs['verbose']

        if len(args) == 1:
            self.load(*args, **kwargs)
        elif 'paths' in kwargs:
            assert len(kwargs['paths']) > 0
            data = None
            for path in tqdm.tqdm_notebook(kwargs.pop('paths')):
                d = Data(path, **kwargs)
                if data is None:
                    data = d
                else:
                    data += d
            self.log = data.log
            self.images = data.images
            self.makeDataArrays()
            self.datadir = data.datadir


    def load(self, datadirOrZippath, MAXDATA=10000000, **kwargs):
        kwargs.setdefault('zeroKeepFraction', 1)
        if datadirOrZippath.endswith('.zip'):
            datadirOrZippath = unzip(datadirOrZippath)
        datadir = getDataDir(datadirOrZippath)
        self.datadir = datadir

        # Load the log data.
        with open(os.path.join(datadir, 'driving_log.csv')) as f:
            logdata = f.read()
        # Check for missing column labels.
        collabels = '''center  left    right   steering    throttle    brake   speed'''.split()
        if not np.all([s in logdata.split('\n')[0] for s in collabels]):
            logdata = ','.join(collabels) + '\n' + logdata
        # Read with Pandas.
        self.log = pd.read_csv(StringIO(logdata))[:MAXDATA]

        # Load the images into arrays.
        if self.verbose: 
            bar = tqdm.tqdm_notebook(total=len(self.log)*3, unit='image')
            def pbar(X):
                for x in X:
                    yield x
                    bar.update()
        else:
            pbar = lambda x: x
        getImg = lambda subPath: misc.imread(os.path.join(datadir, 'IMG', os.path.basename(subPath)))
        sampleimage = getImg(self.log['left'][0])
        self.images = {
            key: np.empty((len(self.log), *sampleimage.shape))
            for key in ('left', 'center', 'right')
        }
        for key in self.images.keys():
            for i, subPath in enumerate(pbar(self.log[key])):
                self.images[key][i, ...] = getImg(subPath)
        self.makeDataArrays()

    def filterSomeZeros(self, zeroKeepFraction=.5):
        if zeroKeepFraction != 1:
            oldsize = len(self)
            keeps = np.random.uniform(size=(len(self),)) < zeroKeepFraction
            keeps = np.logical_or(keeps, self.Y[:, 0] != 0)

            len(self.log[keeps]), sum(keeps), len(self)

            self.log = self.log[keeps]
            for k in 'left', 'center', 'right':
                self.images[k] = self.images[k][keeps]
            self.X = self.X[keeps, ...]
            self.Y = self.Y[keeps, ...]
            newsize = len(self)
            print('Reduced from %d to %d samples.' % (oldsize, newsize))

    def save(self, outpath=None):
        if outpath is None:
            outpath = os.path.join(self.datadir, 'XY.npz')
        if self.verbose:
            with timeit('Saving to %s' % outpath):
                np.savez(outpath, X=self.X, Y=self.Y)
        else:
            np.savez(outpath, X=self.X, Y=self.Y)

    def makeDataArrays(self):
        # Make data arrays.
        self.X = np.vstack(
            [
                self.images[k]
                for k in ('left', 'center', 'right')
            ],
        )
        if self.verbose: print('X is %.3g GB:' % (self.X.size / 2**30,), self.X.shape)

        self.Y = np.vstack([
            np.tile(np.array(y), 3)
            for y in (self.log['steering'], self.log['throttle'], self.log['brake'])
        ]).T

        # Adjust steering angle for side camera views.
        n = len(self.log)
        for j in range(3):
            if j == 0:
                adj = +self.sidecamAdjustment
            elif j == 1:
                adj = 0
            else:
                adj = -self.sidecamAdjustment
            self.Y[j*n:(j+1)*n, 0] += adj
        if self.verbose: print('Y is %.3g KB:' % (self.Y.size / 2**10,), self.Y.shape)

    def __add__(self, other):
        a = self
        b = other
        c = Data(verbose=False)

        # Concatenate data.
        c.log = pd.concat([a.log, b.log], ignore_index=True)
        c.images = {}
        for key in 'left', 'center', 'right':
            c.images[key] = np.vstack([a.images[key], b.images[key]])
        c.makeDataArrays()
        c.datadir = a.datadir

        return c

    def __len__(self):
        return max(len(self.log), len(self.X))

    def frame2time(self, j):
        '''Convert filenames to epoch seconds.'''
        p = self.log['left'][self.log.index[j]].split('left_')[1][:-4]
        y, m, d, h, mi, s, ms =  [int(x) for x in p.split('_')]
        t = datetime.datetime(y, m, d, h, mi, s, ms*1000)
        return (t - EPOCH).total_seconds()


@contextmanager
def timeit(label=None):
    '''Context manager to print elapsed time.
    
    Use like:
    >>> with timeit('waiting'):
    ...     sleep(1.0)
    1.0 sec elapsed waiting.
    '''
    if label is not None:
        print('%s ... ' % label, end='')
    
    s = time.time()
    yield
    e = time.time()
    out = '%.1f sec elapsed.' % (e - s)
    print(out)
