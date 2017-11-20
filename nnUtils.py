"""Utility functions for training Keras networks."""
import tqdm
import time
import tensorflow as tf, keras
import h5py

def isInteractive():
    """Are we in a notebook?"""
    import __main__ as main
    return not hasattr(main, '__file__')


class TensorBoardCallback(keras.callbacks.TensorBoard):
    """Log loss and metrics to TensorBoard for progress monitoring."""
    
    def __init__(self, *args, everyK=1, **kwargs):
        keras.callbacks.TensorBoard.__init__(self, *args, **kwargs)
        self.everyK = everyK
        self.write_batch_performance = True
        self.seen = 0
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance and not (batch % self.everyK):
            for name, value in logs.items():
                if name in ['batch','size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size


class TqdmCallback(keras.callbacks.Callback):
    """Fill a progressbar with batches in terminal or notebook."""
    def __init__(self, nbatch):
        if isInteractive():
            Pbar = tqdm.tqdm_notebook
        else:
            Pbar = tqdm.tqdm
        self.pbar = Pbar(total=nbatch, unit='batch')
    
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.pbar.update()


def loadModel(modelPath):
    """Load a model from HDF5 file."""
    from keras.models import load_model
    from keras import __version__ as keras_version

    # check that model Keras version is same as local Keras version
    f = h5py.File(modelPath, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    return load_model(modelPath)


def fitModelWithDataGenerator(
    model, dataGenerator, modelName,
    epochs=32,
    callbacks=['TqdmCallback', 'TensorBoardCallback'],
    verbose=0,
    **kwargs
    ):
    """Train a Keras model using our DataGenerator object."""
    trainGen = dataGenerator.generate()
    validGen = dataGenerator.generate(validation=True)
    steps_per_epoch = len(trainGen)

    if 'TensorBoardCallback' in callbacks:
        log_dir = '/home/tsbertalan/tensorboardlogs/behavClon/%s-%s/'% (modelName, time.time())
        print('Logging to tensorboard at %s.' % log_dir)
        callbacks[callbacks.index('TensorBoardCallback')] = TensorBoardCallback(
            log_dir=log_dir,
            everyK=10,
        )

    if 'TqdmCallback' in callbacks:
        callbacks[callbacks.index('TqdmCallback')] = TqdmCallback(
            epochs*steps_per_epoch
        )

    return model.fit_generator(
        trainGen,
        steps_per_epoch=steps_per_epoch,
        validation_data=validGen,
        validation_steps=len(validGen),
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs
    )