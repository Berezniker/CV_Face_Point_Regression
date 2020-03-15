from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D,
                                     MaxPooling2D, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import matplotlib.pyplot as plt
from os.path import join, basename
from glob import glob
from time import time, sleep
import numpy as np

MODEL_NAME = 'facepoints_model.hdf5'
COLOR_MODE = 'rgb'
IMAGE_SIZE = (100, 100, 3 if COLOR_MODE == 'rgb' else 1)
NUMBER_OF_FACE_POINTS = 14
FIGURE_SIZE = (15, 10)
BATCH_SIZE = 32
EPOCHS = 100


def show_points(image, face_points):
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')
    for x in range(NUMBER_OF_FACE_POINTS):
        plt.scatter(face_points[2 * x], face_points[2 * x + 1], s=256,
                    marker='$' + str(x) + '$', edgecolors='face', color='r')
    plt.show()
    plt.close()


def save_statistic(model_history):
    N = np.arange(len(model_history.epoch))

    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.plot(N, model_history.history["loss"], label="train_loss")
    axes.plot(N, model_history.history["val_loss"], label="val_loss")
    axes.plot(N, model_history.history["acc"], label="train_acc")
    axes.plot(N, model_history.history["val_acc"], label="val_acc")
    axes.set_title("Training Loss(MSE) and Accuracy")
    axes.set_ylabel("Loss/Accuracy")
    axes.set_xlabel("Epoch #")
    axes.set_yscale('linear')  # 'log'
    axes.legend()

    fig.savefig(fname='history.png')


def load_train_data(face_points_dict, train_img_dir):
    n_images = len(face_points_dict)
    image_data = np.zeros((n_images,) + IMAGE_SIZE, dtype=np.float32)
    face_points = np.zeros((n_images, NUMBER_OF_FACE_POINTS * 2), dtype=np.float64)

    for i, (image_name, face_point) in enumerate(face_points_dict.items()):
        image = img_to_array(load_img(join(train_img_dir, image_name), color_mode=COLOR_MODE)) / 255.
        face_point[0::2] /= image.shape[1] / IMAGE_SIZE[1]
        face_point[1::2] /= image.shape[0] / IMAGE_SIZE[0]
        face_points[i] = face_point
        image_data[i] = resize(image, IMAGE_SIZE)
        # show_points(image_data[i], face_points[i])

    return image_data, face_points


def model_CNN_architecture():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='elu',
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='he_normal', input_shape=IMAGE_SIZE))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, activation='elu',
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(512, activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Dense(NUMBER_OF_FACE_POINTS * 2))

    model.summary()
    # plot_model(model, to_file='./model.png', show_shapes=True)

    return model


def train_detector(face_points_dict, train_img_dir, fast_train=False):
    X, y = load_train_data(face_points_dict, train_img_dir)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=1/5)
    epochs = 1 if fast_train else EPOCHS

    model = model_CNN_architecture()
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    # checkpointer = ModelCheckpoint(filepath=MODEL_NAME, verbose=1, save_best_only=True)
    # tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=15,
    #                               verbose=1, mode='min', min_lr=0.0001)
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        verbose=1,
                        # callbacks=[checkpointer, reduce_lr],
                        validation_data=(X_valid, y_valid),
                        shuffle=True,
                        initial_epoch=0)

    # save_statistic(history)


def load_test_data(dirname):
    file_path = glob(join(dirname, '*.jpg'))
    n_images = len(file_path)
    image_data = np.zeros((n_images,) + IMAGE_SIZE, dtype=np.float32)
    coordinate_compression = np.zeros((n_images, 2), dtype=np.float64)

    for i, image_path in enumerate(file_path):
        image = img_to_array(load_img(image_path, color_mode=COLOR_MODE)) / 255.
        coordinate_compression[i] = np.array([image.shape[1] / IMAGE_SIZE[1],
                                              image.shape[0] / IMAGE_SIZE[0]])
        image_data[i] = resize(image, IMAGE_SIZE)
    file_names = [basename(file_name) for file_name in file_path]

    return image_data, file_names, coordinate_compression


def detect(model, test_img_dir):
    data, image_names, compression_ratios = load_test_data(test_img_dir)
    detected_points = model.predict(data)
    dictionary_of_points = {image_name: points * np.tile(ratio, NUMBER_OF_FACE_POINTS)
                            for image_name, points, ratio in
                            zip(image_names, detected_points, compression_ratios)}

    # CHECK
    for img_name in glob(join(test_img_dir, '*.jpg')):
        show_points(img_to_array(load_img(img_name)) / 255.,
                    dictionary_of_points[basename(img_name)])
        sleep(3)

    return dictionary_of_points


if __name__ == '__main__':
    gt_dir = 'tests\\00_test_img_input\\train\\gt.csv'
    train_dir = 'tests\\00_test_img_input\\train\\images'
    gt_out_dir = 'tests\\00_test_img_check\\gt\\gt.csv'
    test_dir = 'tests\\00_test_img_input\\test\\images'

    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            next(fhandle)
            for line in fhandle:
                parts = line.rstrip('\n').split(',')
                coords = np.array([float(x) for x in parts[1:]], dtype='float64')
                res[parts[0]] = coords
        return res

    # FAST TRAIN
    # print('[DATA] Training ...')
    # start_time = time()
    # train_gt = read_csv(gt_dir)
    # train_detector(train_gt, train_dir, fast_train=True)
    # print('time:', time() - start_time)

    # CRASH DETECTING
    # print('[INFO] Detecting ...')
    # start_time = time()
    # model = load_model(MODEL_NAME)
    # X_test = detect(model, r'./crash_test')
    # print('time:', time() - start_time)


    model = Sequential()

    model.add(Conv2D(filters=48, kernel_size=(3, 3), input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=192, kernel_size=(7, 7)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Dropout(rate=0.3))

    model.add(Dense(1024))
    model.add(Dropout(rate=0.4))

    model.add(Dense(20))

    model.summary()