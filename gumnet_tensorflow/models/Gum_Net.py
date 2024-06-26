import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input, Conv3D, Dense, Flatten, Concatenate, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from aitom.align.deep.gum.utils import get_initial_weights, correlation_coefficient_loss, alignment_eval
from aitom.align.deep.gum.FeatureCorrelation import FeatureCorrelation
from aitom.align.deep.gum.FeatureL2Norm import FeatureL2Norm
from aitom.align.deep.gum.RigidTransformation3DImputation import RigidTransformation3DImputation
from aitom.align.deep.gum.SpectralPooling import SpectralPooling

# define Gumnet model structure
def GUM(img_shape=None):
    if img_shape is None:
        img_shape = [32, 32, 32]
    input_shape = (img_shape[0], img_shape[1], img_shape[2], 1)
    channel_axis = -1

    # feature extractors share the same weights
    shared_conv1 = Conv3D(32, (3, 3, 3), padding='valid')
    shared_conv2 = Conv3D(64, (3, 3, 3), padding='valid')
    shared_conv3 = Conv3D(128, (3, 3, 3), padding='valid')
    shared_conv4 = Conv3D(256, (3, 3, 3), padding='valid')
    shared_conv5 = Conv3D(512, (3, 3, 3), padding='valid')

    # feature extraction for s_b
    main_input = Input(shape=input_shape, name='main_input')
    # shared convolutional layer
    # ReLU activation function is used to add nonlinearity
    # Batch Normalization is applied to help the model training be more stable
    # Spectral pooling: reference to models/layers/SpectralPooling.py
    # FeatureL2Norm: reference to models/layers/FeatureL2Norm.py
    v_a = shared_conv1(main_input)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = SpectralPooling((26, 26, 26), (22, 22, 22))(v_a)

    v_a = shared_conv2(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = SpectralPooling((18, 18, 18), (15, 15, 15))(v_a)

    v_a = shared_conv3(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = SpectralPooling((12, 12, 12), (10, 10, 10))(v_a)

    v_a = shared_conv4(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = SpectralPooling((8, 8, 8), (7, 7, 7))(v_a)

    v_a = shared_conv5(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = FeatureL2Norm()(v_a)

    # feature extraction for s_a
    # Defines the auxiliary input to the model, which is used to process the second sub-image.
    auxiliary_input = Input(shape=input_shape, name='aux_input')

    v_b = shared_conv1(auxiliary_input)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = SpectralPooling((26, 26, 26), (22, 22, 22))(v_b)

    v_b = shared_conv2(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = SpectralPooling((18, 18, 18), (15, 15, 15))(v_b)

    v_b = shared_conv3(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = SpectralPooling((12, 12, 12), (10, 10, 10))(v_b)

    v_b = shared_conv4(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = SpectralPooling((8, 8, 8), (7, 7, 7))(v_b)

    v_b = shared_conv5(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = FeatureL2Norm()(v_b)

    # correlation layer
    c_ab = FeatureCorrelation()([v_a, v_b])
    c_ab = FeatureL2Norm()(c_a0b)

    # correlation layer
    c_ba = FeatureCorrelation()([v_b, v_a])
    c_ba = FeatureL2Norm()(c_ba)

    c_ab = Conv3D(1024, (3, 3, 3))(c_ab)
    c_ab = BatchNormalization(axis=channel_axis)(c_ab)
    c_ab = Activation('relu')(c_ab)

    c_ab = Conv3D(1024, (3, 3, 3))(c_ab)
    c_ab = BatchNormalization(axis=channel_axis)(c_ab)
    c_ab = Activation('relu')(c_ab)

    c_ab = Flatten()(c_ab)

    c_ba = FeatureL2Norm()(c_ba)
    c_ba = Conv3D(1024, (3, 3, 3))(c_ba)
    c_ba = BatchNormalization(axis=channel_axis)(c_ba)
    c_ba = Activation('relu')(c_ba)

    c_ba = Conv3D(1024, (3, 3, 3))(c_ba)
    c_ba = BatchNormalization(axis=channel_axis)(c_ba)
    c_ba = Activation('relu')(c_ba)

    c_ba = Flatten()(c_ba)

    c = Concatenate()([c_ab, c_ba])

    c = Dense(2000)(c)
    c = Dense(2000)(c)

    weights = get_initial_weights(2000)

    # estimated 3D rigid body transformation parameters
    c = Dense(6, weights=weights)(c)
    c = Activation('sigmoid')(c)

    mask_1 = Input(shape=input_shape, name='mask_1')
    mask_2 = Input(shape=input_shape, name='mask_2')

    x, mask1, mask2 = RigidTransformation3DImputation(
        (img_shape[0], img_shape[1], img_shape[2]))([main_input, auxiliary_input, mask_1, mask_2, c])

    model = Model(inputs=[main_input, auxiliary_input, mask_1, mask_2], outputs=x)

    adam = Adam()
    model.compile(loss=correlation_coefficient_loss, optimizer=adam)

    return model


def get_transformation_output_from_model(model, x_train, y_train, mask_1, mask_2):
    layer_output = np.zeros((x_train.shape[0], 6))
    get_layer_output_func = K.function(
        [model.layers[0].input, model.layers[1].input, model.layers[-3].input,
         model.layers[-4].input],
        [model.layers[-2].output])

    layer_output = get_layer_output_func([x_train, y_train, mask_1, mask_2])[0]

    return layer_output