import os.path
import keras, numpy as np, matplotlib.pyplot as plt
import nnUtils, loadData, models


## CREATE DATA GENERATOR.
datadir = os.path.join(os.path.expanduser('~'), 'data2', 'behavioralCloning')
dataGenerator = loadData.DeemphasizedZeroDataGenerator(
    [
        os.path.join(loadData.HOME, 'data2', 'behavioralCloning', p)
        for p in (
            'mouseForward.zip',
            'mouseReverse.zip',
            'longCarefulForward.zip',
            'longCarefulReverse.zip',
            # 'data_provided.zip',
            # 'dirtSidesForward.zip',
            # 'dirtSidesReverse.zip',
        )
    ],
)


## DEFINE MODEL.
epochs = 6
learningRate = .0001
kernel_regularizer = keras.regularizers.l2(0.0001)
#bias_regularizer = keras.regularizers.l2(0.01)
bias_regularizer = None
modelname = (
    'deemZero-mouse_longCare-kl2%.1g-%depoch-subsamp%.1g' % (
    kernel_regularizer.l2, epochs, epochSubsampling
)).replace('0.', 'p')

model = models.Nvidia(
    len(dataGenerator.responseKeys), dataGenerator.sampleImageShape,
    optimizer=keras.optimizers.Nadam(
        lr=learningRate,
    ),
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
)


## TRAIN THE MODEL.
trainGen = dataGenerator.generate(epochSubsampling=1)
validGen = dataGenerator.generate(validation=True)
history = nnUtils.fitModelWithDataGenerator(
    model, dataGenerator, modelname, 
    epochs=epochs, max_queue_size=100,
)

# Save the trained model.
fpath = os.path.join(datadir, '%s.h5' % modelname)
print('Saving to', fpath)
model.save(fpath)


## PLOT TRAIN/VALIDATE COMPARISON.
trainIndices, validationIndices = dataGenerator._indices
for ind, lab in zip(dataGenerator._indices, ('train', 'validation')):
    
    imgs = []
    responses = []
    ind = np.copy(ind)
    ind.sort()
    for k in range(ind[0], ind[0]+1000):
        try:
            image, response = dataGenerator.getImageResponse(k)
            imgs.append(image.reshape((1, *image.shape)))
            responses.append(response)
        except KeyError:
            break

    # Get data, true responses, and [smoothed] predictions.
    X = np.vstack(imgs)
    Y = np.vstack(responses)
    pred = model.predict(X)
    from collections import deque
    def runningMean(x, N):
        y = []
        hist = deque(maxlen=N)
        for a in x:
            hist.append(a)
            y.append(np.mean(hist))
        return np.array(y)
    filtersize = 6
    scale = 1
    smoothed = scale * runningMean(pred, filtersize)
    
    fig, ax = plt.subplots()
    if lab == 'train':
        start = 580
        end = start + 160
    else:
        start = 243
        end = start + 60
    samples = range(start, end)
    
    ax.plot(samples, Y[start:end], label=r'truth $\theta$', color='black')
    ax.plot(
        samples, 
        pred[start:end],
        label=r'predictions $\hat\theta$', color='magenta',
    )
    ax.plot(
        samples, smoothed[start:end], 
        label=r'$\hat\rho = %.1f \cdot box_{%d}(\hat\theta)$' % (scale, filtersize),
        #alpha=.5,
        color='green',
    )
    
    # Add some insets of the input.
    ninset =6
    w = h = 1./(ninset+2)
    for xloc, i in zip(
        np.linspace(w, 1-2*w, ninset),
        np.linspace(start, end-1, ninset).astype(int)
    ):
        l = xloc; b = .1100
        inset = fig.add_axes([l, b, w, h])
        loadData.showxy(
            X[i], Y[i], y2=pred[i], ax=inset, 
            l1kwargs=dict(color='white', linewidth=2), l2kwargs=dict(color='magenta', linewidth=2, alpha=.5)
        )
        inset.set_title(
            '$t=%d$' % samples[i-start]
            ,fontsize=10,
        )
    
    ax.set_xlabel('time [samples]')
    ax.set_ylabel('angle [radians]')
    ax.set_title(lab)
    ax.legend(loc='upper left');
    for fpath in [
        'doc/smoothingEffect-%s-%s.png' % (modelname, lab),
        'doc/smoothingEffect-%s.png' % lab,
    ]:
        fig.savefig(fpath)
    
plt.show()
