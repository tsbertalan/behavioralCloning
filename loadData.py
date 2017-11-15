from contextlib import contextmanager
import datetime, time, os
import matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np
import pandas as pd
from pandas.compat import StringIO
from scipy import misc
import tqdm
import sklearn.utils  # shuffle

HOME = os.path.expanduser('~')
EPOCH = datetime.datetime(1970, 1, 1)


def unzip(archivepath, outdir='/tmp/data/', verbose=True):
    do = lambda: os.system('unzip -nq "%s" -d "%s"' % (archivepath, outdir))
    if verbose:
        with timeit('Unzipping %s' % archivepath):
            do()
    else:
        do()
    return outdir


def getDataDir(zipOutDir):
    dirs = os.listdir(zipOutDir)
    if 'IMG' not in dirs:
        assert len(dirs) > 0
        zipOutDir = os.path.join(zipOutDir, dirs[0])
    return zipOutDir

_LCR = ('left', 'center', 'right')

class DataGenerator(object):

    def __init__(self, 
        zipPaths, 
        sidecamAdjustment=.05, verbose=True, validationFraction=.2, batchBaseSize=32,
        responseKeys=('steering',),
        shuffleBatch=True,
        ):

        # Set attributes.
        self.sidecamAdjustment = sidecamAdjustment
        self.verbose = verbose
        self.batchBaseSize = batchBaseSize
        self.responseKeys = responseKeys
        self.validationFraction = validationFraction
        self.shuffleBatch = shuffleBatch

        # Unzip the archives and load all the log files.
        if isinstance(zipPaths, str):
            zipPaths = [zipPaths]
        assert len(zipPaths) > 0
        self.log = None
        self.subLogLengths = []
        for path in zipPaths:
            # Unzip the zipfile.
            datadir = getDataDir(unzip(path, verbose=verbose))

            # Load the log data.
            with open(os.path.join(datadir, 'driving_log.csv')) as f:
                logdata = f.read()

            # Add missing column labels.
            collabels = """center  left    right   steering    throttle    brake   speed""".split()
            lds = logdata.split('\n')[0]
            if not np.all([s in lds for s in collabels]):
                logdata = ','.join(collabels) + '\n' + logdata

            # Read with Pandas.
            newLog = pd.read_csv(StringIO(logdata))
            self.subLogLengths.append(len(newLog))

            # Make image paths absolute.
            for key in _LCR:
                for i in range(len(newLog)):
                    v = newLog.at[i, key]
                    newLog.at[i, key] = os.path.join(datadir, 'IMG', os.path.basename(v))

            # Append the log.
            if self.log is None:
                self.log = newLog
            else:
                self.log = pd.concat([self.log, newLog], ignore_index=True)

    def shuffle(self):
        # Take last part of each sublog as a validation set.
        # Taking a random subset doesn't work as well since validation
        # samples might be very similar to adjacent training samples.
        validationIndices = []
        trainIndices = []
        start = 0
        for sublen in self.subLogLengths:
            end = start + sublen
            nvalid = int(sublen * self.validationFraction)
            trainIndices.append(
                np.arange(start, end-nvalid)
            )
            validationIndices.append(
                np.arange(end-nvalid, end)
            )
            start += sublen
        trainIndices = np.vstack(trainIndices)
        validationIndices = np.vstack(validationIndices)
        for ind in trainIndices, validationIndices:
            np.random.shuffle(ind)
            print(ind[:10])
        
        self.__indices = [trainIndices, validationIndices]

        # Set index state for the generator.
        self.__state = [0, 0]

    @property
    def _indices(self):
        try:
            return self.__indices
        except AttributeError:
            self.shuffle()
            return self.__indices

    @property
    def _state(self):
        try:
            return self.__state
        except AttributeError:
            self.shuffle()
            return self.__state

    def getImageResponse(self, rawIndex, key='center'):
        """For debugging purposes."""
        imgPath = self.log.at[rawIndex, key]
        response = [self.log.at[rawIndex, key] for key in self.responseKeys]
        return misc.imread(imgPath), response

    def sampleRow(self, validation=False):
        # Get the index and increment the state.
        indices = self._indices[validation]
        state = self._state[validation]
        j = indices[state]
        
        # Reset or increment the counter.
        if self._state[validation] == len(indices) - 1:
            self._state[validation] = 0
        else:
            self._state[validation] += 1

        # Get the left, center, and right image paths
        # and corresponding requested response variables.
        lcrImagePaths = [self.log.at[j, key] for key in _LCR]
        response = np.array([self.log.at[j, key] for key in self.responseKeys])
        return lcrImagePaths, response

    rowsPerSample = 3
    def sample(self, validation=False):
        lcrImagePaths, response = self.sampleRow(validation=validation)
        X = []
        Y = []
        for icam, imagePath in enumerate(lcrImagePaths):
            x = misc.imread(imagePath)
            x = x.reshape((1, *x.shape))
            y = np.copy(response)
            # Make the steering angle adjustment for side cameras.
            y[0] += [self.sidecamAdjustment, 0, -self.sidecamAdjustment][icam]
            X.append(x)
            Y.append(y)

        if self.shuffleBatch: X, Y = sklearn.utils.shuffle(X, Y)

        return np.vstack(X), np.vstack(Y)

    @property
    def sampleImageShape(self):
        if not hasattr(self, '_sampleImageShape'):
            self._sampleImageShape = misc.imread(self.log.at[0, 'center']).shape
        return self._sampleImageShape

    def sampleBatch(self, validation=False):

        state = self._state[validation]
        samplesInBatch = min(self.batchBaseSize, self.len(validation) - state)

        X = np.empty((samplesInBatch * self.rowsPerSample, *self.sampleImageShape))
        Y = np.empty((samplesInBatch * self.rowsPerSample, len(self.responseKeys)))

        for j in range(samplesInBatch):
            x, y = self.sample(validation=validation)
            X[j*self.rowsPerSample:(j+1)*self.rowsPerSample, ...] = x
            Y[j*self.rowsPerSample:(j+1)*self.rowsPerSample, ...] = y

        return X, Y

    def len(self, validation=False):
        return len(self._indices[validation])

    def generate(self, validation=False, stopOnEnd=False, epochSubsampling=1.0):
        """Emit a generator that yields data for training.

        epochSubsampling : float, optional
            For faster training, this number can be artifically lowered,
            causing faster epoch passage. Ensuing shuffling every "epoch" ensures that
            all samples still have an equal chance of being seen.

        """
        self.shuffle()
        assert 0 < epochSubsampling <= 1.0

        import collections
        class BatchGenerator(collections.Iterator):
            def __init__(self):
                self._state = 0

            def __len__(innerSelf):
                """Return the expected number of batches required to use "all" the data."""
                return int(epochSubsampling * self.nbatches(validation))

            def __next__(innerSelf):
                if innerSelf._state == len(innerSelf):
                    if not validation:
                        self.shuffle()
                    innerSelf._state = 0
                    if stopOnEnd:
                        raise StopIteration
                else:
                    innerSelf._state += 1
                return self.sampleBatch(validation=validation)

            def __iter__(innerSelf):
                return innerSelf

        return BatchGenerator()

    def nbatches(self, validation=False):
        return np.math.ceil(self.len(validation) / self.batchBaseSize)

    # @property
    # def batchSize(self):
    #     return self.batchBaseSize * self.rowsPerSample

    def __len__(self):
        return self.rowsPerSample * len(self.log)


@contextmanager
def timeit(label=None):
    """Context manager to print elapsed time.
    
    Use like:
    >>> with timeit('waiting'):
    ...     sleep(1.0)
    1.0 sec elapsed waiting.
    """
    if label is not None:
        print('%s ... ' % label, end='')
    
    s = time.time()
    yield
    e = time.time()
    out = '%.1f sec elapsed.' % (e - s)
    print(out)