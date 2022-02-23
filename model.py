from keras.applications.inception_v3 import InceptionV3
from keras.applications import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
import os
import sys
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge

inputshape = (256, 256, 3)
load_imagenet = True

#create the base pre-trained model
def Net(load_imagenet=load_imagenet):
    if load_imagenet:
        base_model = NASNetLarge(weights='imagenet', include_top=False)
    else:
        base_model = NASNetLarge(weights=None, include_top=False, input_shape=inputshape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = True
    return model

model = Net()
model.summary()