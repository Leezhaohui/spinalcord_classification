import keras.backend as K

def acc(y_true, y_pred): # (batchsize, 2)
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    tp = K.sum(y_true * y_pred)
    tn = K.sum((1-y_true) * (1-y_pred))
    fp = K.sum((1-y_true) * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    return (tp + tn) / (tp + tn + fp + fn)

def sen(y_true, y_pred): # (batchsize, 2)
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1-y_pred))
    return (tp) / (tp + fn)

def spe(y_true, y_pred): # (batchsize, 2)
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    tn = K.sum((1-y_true) * (1-y_pred))
    fp = K.sum((1-y_true) * y_pred)
    return (tn) / (tn + fp)
