from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Reshape, Activation, BatchNormalization
from keras.models import Model

from trainbench.keras.callbacks import SimpleCallback

parameters = {
    'optimizer': [

    ],
    'fc2_size': [50, 100, 256, 512]

}


def train(parameters):
    batch_size = 32

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        '/data/full/train/',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    # validation
    validation_generator = train_datagen.flow_from_directory(
        '/data/full/validation/',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    # model
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))

    input_tensor = vgg16.input
    # vgg16 goes here
    fc1 = Conv2D(100, (7, 7))(vgg16.output)
    fc1_bn = BatchNormalization()(fc1)
    fc1_act = Activation('relu')(fc1_bn)

    fc2 = Conv2D(2, (1, 1))(fc1_act)
    fc2_bn = BatchNormalization()(fc2)
    fc2_act = Activation('softmax')(fc2_bn)

    output_tensor = Reshape((2,))(fc2_act)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.fit_generator(
        train_generator,
        steps_per_epoch=(3000 / 32),
        verbose=1,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=10)
