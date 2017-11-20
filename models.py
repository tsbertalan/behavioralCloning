"""Model constructor functions."""
import keras

def prod(it):
    """Like the sum() builtin, but a product."""
    out = 1
    for x in it:
        try:
            out *= float(x)
        except TypeError:
            out *= int(x)
    return out

def InceptionV3(
    nout, input_shape,
    doCompile=True, 
    loss='mse', optimizer='nadam', metrics=['accuracy', 'mae'],
    ):
    
    # Normalize.
    x = keras.layers.Input(input_shape)
    x = keras.layers.Lambda(lambda y: (y / 255.0) - 0.5)(x)
    
    model = keras.applications.inception_v3.InceptionV3(
        include_top=False, 
        weights='imagenet', 
        input_tensor=x,
        #input_shape=input_shape, 
        #pooling=None, 
        #classes=1000
    )
    img_input = model.layers[0].input
    nUnfrozen = sum([prod(w.shape) for w in model.trainable_weights])
    for layer in model.layers:
        layer.trainable = False
    
    # Get flattened output.
    x = model.layers[-1].output
    print('Pretrained model provides {:,} features.'.format(prod(x.shape[1:])))
    x = keras.layers.Flatten()(x)
        
    # Add layers.
    init = dict(bias_initializer='zeros', kernel_initializer='glorot_normal')
    x = keras.layers.Dense(16, activation='relu', **init)(x)
    for k in range(4):
        x = keras.layers.Dense(16, activation='relu', **init)(x)
    x = keras.layers.Dense(nout, activation='tanh', name='tanh')(x)
    
    # Make and compile model.
    model = keras.Model(inputs=img_input, outputs=x)
    
    nFrozen = sum([prod(w.shape) for w in model.trainable_weights])
    print("Freezing reduces trainable size from {:,} to {:,} parameters.".format(nUnfrozen, nFrozen))
    
    if doCompile:
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
    return model
 
# (from top, from bottom), (from left, from right)
cropping = ((52, 18), (0, 0))


def Nvidia(
    nout, input_shape,
    doCompile=True,
    loss='mse', optimizer='nadam', metrics=['accuracy', 'mae'],
    kernel_regularizer=None, bias_regularizer=None,
    convActivation='relu', denseActivation='relu', lastActivation='tanh'
    ):
    """A simpler CNN architecture adapted from http://arxiv.org/abs/1604.07316"""

    # Normalize.
    x = img_input = keras.layers.Input(input_shape)
    x = keras.layers.Cropping2D(
        cropping=cropping,
    )(x)
    x = keras.layers.Lambda(lambda y: (y / 255.0) - 0.5)(x)

    # Convolutional layers
    convKwargs = dict(
        kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer,
        activation=convActivation,
    )
    x = keras.layers.Conv2D(24, 5, strides=(2,2), **convKwargs)(x)
    x = keras.layers.Conv2D(36, 5, strides=(2,2), **convKwargs)(x)
    x = keras.layers.Conv2D(48, 5, strides=(3,3), **convKwargs)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1,1), **convKwargs)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1,1), **convKwargs)(x)
    print('Convolutional part provides {:,} features.'.format(prod(x.shape[1:])))
    x = keras.layers.Flatten()(x)

    # Fully-connected layers.
    fcKwargs = dict(
        activation=denseActivation,
    )
    x = keras.layers.Dense(100, **fcKwargs)(x)
    x = keras.layers.Dense(50, **fcKwargs)(x)
    x = keras.layers.Dense(10, **fcKwargs)(x)
    fcKwargs['activation'] = lastActivation
    x = keras.layers.Dense(nout, **fcKwargs)(x)

    # Make and compile model.
    model = keras.Model(inputs=img_input, outputs=x)

    nparam = sum([prod(w.shape) for w in model.trainable_weights])
    print("Model has {:,} trainable parameters.".format(nparam))
    
    if doCompile:
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
    return model
