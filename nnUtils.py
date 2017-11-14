import tqdm
import time
import tensorflow as tf, keras
import h5py

class TensorBoardCallback(keras.callbacks.TensorBoard):
    
    def __init__(self, *args, **kwargs):
        keras.callbacks.TensorBoard.__init__(self, *args, **kwargs)
        self.write_batch_performance = True
        self.seen = 0
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance == True:
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
    def __init__(self, nbatch):
        self.pbar = tqdm.tqdm_notebook(total=nbatch, unit='epoch')
    
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.pbar.update()


def loadModel(modelPath):
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
    verbose=1,
    **kwargs
    ):
    trainGen = dataGenerator.generate()
    validGen = dataGenerator.generate(validation=True)

    if 'TqdmCallback' in callbacks:
        callbacks[callbacks.index('TqdmCallback')] = TqdmCallback(epochs)

    if 'TensorBoardCallback' in callbacks:
        log_dir = '/home/tsbertalan/tensorboardlogs/behavClon/%s-%s/'% (modelName, time.time())
        callbacks[callbacks.index('TensorBoardCallback')] = TensorBoardCallback(
            log_dir=log_dir,
        )

    return model.fit_generator(
        trainGen,
        steps_per_epoch=len(trainGen),
        validation_data=validGen,
        validation_steps=len(validGen),
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        **kwargs
    )