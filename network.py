from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Activation, Flatten, BatchNormalization, AveragePooling2D, Add
import tensorflow as tf
from tensorflow.keras.models import Model

def build_model(stack_depth, img_height, img_width, output_size):
    model = Sequential()

    model.add(Conv2D(8, (4, 4), input_shape = (img_height, img_width, stack_depth), padding="valid", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (4, 4), input_shape = (img_height, img_width, stack_depth), padding="valid", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding="same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding="same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), padding="same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), padding="same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(192, activation="relu"))
    model.add(Dense(96, activation="relu"))
    model.add(Dense(96, activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))

    adam = Adam(learning_rate=1e-5)

    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

    return model

def build_model_resnet(stack_depth, img_height, img_width, output_size):
    def residual_block(X, downsample, kernel_size, n_filters):
        strides = 1
        if downsample:
            strides = 2
        y = Conv2D(n_filters, kernel_size = kernel_size, strides = strides, padding="same", activation = "relu")(X)
        y = BatchNormalization()(y)
        y = Conv2D(n_filters, kernel_size = kernel_size, strides = 1, padding="same", activation = "relu")(y)
        y = BatchNormalization()(y)
        if downsample:
            X = Conv2D(n_filters, kernel_size = 1, strides = 2, padding="same", activation = "relu")(X)
        out = Add()([X, y])
        out = Activation("relu")(out)

        return out

    input = Input(shape = (img_height, img_width, stack_depth))
    # input = BatchNormalization()(input)

    n_filters = 16

    t = Conv2D(n_filters, kernel_size = 3, strides = 1, padding="same", activation = "relu")(input)
    t = BatchNormalization()(t)

    residual_blocks = [2, 5, 3]

    for i in range(len(residual_blocks)):
        num_blocks = residual_blocks[i]
        for j in range(num_blocks):
            downsample = (j == 0 and i != 0)
            t = residual_block(t, downsample = downsample, kernel_size = 3, n_filters=n_filters)
        n_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)

    fcl = Dense(192, activation="relu")
    fcl = Dense(96, activation="relu")
    fcl = Dense(96, activation="relu")

    output = Dense(output_size, activation='softmax')(t)

    model = Model(input, output)

    adam = Adam(learning_rate=1e-5)

    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

    return model

def limit_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], False)


def load_model(name):
    return tf.keras.models.load_model(name)