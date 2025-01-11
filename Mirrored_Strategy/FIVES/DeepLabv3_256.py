from atrous import *
from swin import *

import keras
from keras import backend as K
from tensorflow.keras.models import Sequential, Model

IMAGE_SIZE = 256

def create_model():
    InputL = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="InputLayer")

    patches = Patches(patch_size)(InputL)
    # print(patches.shape)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim) (patches) 
    print(x.shape)
    x = SwinTransformer(
        dim=embed_dim, #64
        num_patch=(num_patch_x, num_patch_y), #16, 16
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    # print(x.shape)
    patch_merging_layer = PatchMerging(patch_size)
    merged_image = patch_merging_layer(x)

    #print("merged:", merged_image.shape)

    x = Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same')(merged_image)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    backbone = Xception(include_top=False, weights='imagenet', input_tensor=InputL)
    # DCNN = backbone.get_layer('block8_sepconv2_act').output

    print(x.shape)

    ASPP = AtrousSpatialPyramidPooling(x)
    ASPP = UpSampling2D(size=(IMAGE_SIZE//2//ASPP.shape[1], IMAGE_SIZE//2//ASPP.shape[2]), name="AtrousSpatial")(ASPP)
    print("ASPP",ASPP.shape)

    LLF = backbone.get_layer('block1_conv1_act').output
    LLF = ZeroPadding2D(padding=((0, 1), (0, 1)))(LLF)
    LLF = ConvBlock(filters=128, kernel_size=1,name="LLF-ConvBlock")(LLF)

    print(LLF.shape)

    # Combined
    combined = Concatenate(axis=-1, name="Combine-LLF-ASPP")([ASPP, LLF])
    features = ConvBlock(name="Top-ConvBlock-1")(combined)
    features = ConvBlock(name="Top-ConvBlock-2")(features)
    upsample = UpSampling2D(size=(IMAGE_SIZE//features.shape[1], IMAGE_SIZE//features.shape[1]), interpolation='bilinear', name="Top-UpSample")(features)

    # Output Mask
    PredMask = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False, name="OutputMask")(upsample)

    model = Model(InputL, PredMask, name="DeepLabV3-Plus")
    model.summary()
    return model




