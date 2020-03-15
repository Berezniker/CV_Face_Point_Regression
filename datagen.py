from albumentations import Compose, ShiftScaleRotate, KeypointParams
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle, count
from random import random
import numpy as np

NUMBER_OF_FACE_POINTS = 14
FIGURE_SIZE = (15, 10)


class DataGen(object):
    def __init__(self, X, y, n_points=NUMBER_OF_FACE_POINTS,
                 flip=True, rotate=True, shift=True, scale=True,
                 rotate_limit=15, shift_limit=0.05, scale_limit=0.05):
        self.X = X
        self.y = y
        self.n_points = n_points
        self.transform = {'flip': flip, 'rotate': rotate,
                          'shift': shift, 'scale': scale}
        self.limit = {'rotate': rotate_limit, 'shift': shift_limit,
                      'scale': scale_limit}

    def generate_data(self):
        yield from cycle(zip(self.X, self.y))

    def create_transformer(self, image, points):
        return Compose([ShiftScaleRotate(shift_limit=self.limit['shift'] if self.transform['shift'] else 0,
                                         scale_limit=self.limit['scale'] if self.transform['scale'] else 0,
                                         rotate_limit=self.limit['rotate'] if self.transform['rotate'] else 0,
                                         interpolation=1,
                                         border_mode=0,
                                         p=14 / 15)],
                       keypoint_params=KeypointParams(format='xy'),
                       p=1
                       )(image=image, keypoints=points)

    def flow(self, batch_size=100):
        X_batch = np.empty((batch_size,) + self.X.shape[1:])
        y_batch = np.empty((batch_size,) + self.y.shape[1:])
        fliplr_coord = [6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 16, 17, 14, 15,
                        12, 13, 10, 11, 8, 9, 20, 21, 26, 27, 24, 25, 22, 23]
        data = self.generate_data()

        while True:
            i = 0
            while i < batch_size:
                X, y = next(data)
                if np.min(y) <= 0 or np.max(y) >= max(X.shape[:2]):
                    continue
                transformed = self.create_transformer(X, y.reshape(self.n_points, 2))
                X, y = transformed['image'], np.array(transformed['keypoints']).ravel()
                if y.size < self.n_points * 2:
                    continue
                if self.transform['flip'] and random() < 1/2:
                    X = np.fliplr(X)
                    y[0::2] = X.shape[1] - y[0::2]
                    y = y[fliplr_coord]

                # self.visualize(X, y)
                X_batch[i, ...], y_batch[i, ...] = X, y
                i += 1

            yield (X_batch, y_batch)

    def visualize(self, image, points):
        fig = plt.figure(figsize=FIGURE_SIZE)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.axis('off')
        for x in range(self.n_points):
            plt.scatter(points[2 * x], points[2 * x + 1], s=256,
                        marker='$' + str(x) + '$', edgecolors='face', color='r')
        plt.show()
        plt.close(fig=fig)


if __name__ == '__main__':
    image = np.load('./tests/00_test_img_input/train/train_dataset_grayscale.npy')
    points = np.load('./tests/00_test_img_input/train/train_labels.npy')

    X_train, X_val, y_train, y_val = train_test_split(image, points, test_size=1/5, random_state=13)

    # visualize ON
    # for _ in DataGen(image, points).flow(1):
    #     pass

    # model = load_model(filepath='facepoints_model.hdf5')
    # model.fit_generator(DataGen(X_train, y_train).flow(batch_size=20),
    #                     steps_per_epoch=200,
    #                     epochs=5,
    #                     verbose=2,
    #                     validation_data=(X_val, y_val))
