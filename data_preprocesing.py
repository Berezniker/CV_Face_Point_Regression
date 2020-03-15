from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize, rotate
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imsave
from math import cos, sin
from random import sample
from os.path import join
from time import time
import numpy as np

IMAGE_SIZE = (100, 100, 1)
NUMBER_OF_FACE_POINTS = 14
FIGURE_SIZE = (15, 10)
n_img = -1


def show_points(image, face_points):
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')
    for x in range(NUMBER_OF_FACE_POINTS):
        plt.scatter(face_points[2 * x], face_points[2 * x + 1], s=256,
                    marker='$' + str(x) + '$', edgecolors='face', color='r')
    plt.show()
    plt.close()


def save_points(image, face_points):

    def get_current_file_name():
        global n_img  # n_img = -1
        n_img += 1
        return '{0:0{width}}.jpg'.format(n_img, width=5)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')
    for x in range(NUMBER_OF_FACE_POINTS):
        plt.scatter(face_points[2 * x], face_points[2 * x + 1], s=256,
                    marker='$' + str(x) + '$', edgecolors='face', color='r')

    plt.savefig('tests\\00_test_img_input\\train\\labeled_images\\' + get_current_file_name(), bbox_inches='tight')
    plt.close()


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def data_augmentation(face_points_dict, img_dir, dirname,
                      target_size=IMAGE_SIZE, color_mode='grayscale',
                      rotates=True, flips=True, shuffle_image=True):

    def load_data():
        nonlocal face_points_dict, img_dir, target_size, color_mode
        n_images = len(face_points_dict)
        image_data = np.zeros((n_images,) + target_size, dtype=np.float32)
        face_points = np.zeros((n_images, NUMBER_OF_FACE_POINTS * 2), dtype=np.float64)

        for i, (image_name, face_point) in enumerate(face_points_dict.items()):
            image = img_to_array(load_img(join(img_dir, image_name), color_mode=color_mode)) / 255.
            compression_ratios = np.array([image.shape[0] / target_size[0],
                                           image.shape[1] / target_size[1]])
            face_points[i] = face_point / np.tile(compression_ratios, NUMBER_OF_FACE_POINTS)
            image_data[i] = resize(image, target_size)
            # show_points(np.squeeze(image_data[i]), face_points[i])

        return image_data, face_points

    def save_csv(new_face_points, filename='./gt.csv'):
        with open(filename, mode='w') as fhandle:
            print('filename,'
                  'x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,'
                  'x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14',
                  file=fhandle)
            for filename in sorted(new_face_points.keys()):
                points_str = ','.join(map(str, new_face_points[filename]))
                print('%s,%s' % (filename, points_str), file=fhandle)

    n_img = 0
    csv_points = {}
    fliplr_coord = [6, 7, 4, 5, 2, 3, 0, 1, 18, 19, 16, 17, 14, 15,
                    12, 13, 10, 11, 8, 9, 20, 21, 26, 27, 24, 25, 22, 23]
    images_dirname = join(dirname, 'images')

    def get_current_file_name():
        nonlocal n_img
        return '{0:0{width}}.jpg'.format(n_img, width=6)

    def save_image(image):
        nonlocal n_img, images_dirname
        imsave(join(images_dirname, get_current_file_name()),
               (image * 255).astype(np.uint8))
        n_img += 1

    def flip_coord(image_size, points):
        nonlocal fliplr_coord
        flip_points = np.zeros_like(points)
        flip_points[0::2] = image_size[0] - points[0::2]
        flip_points[1::2] = points[1::2]
        return flip_points[fliplr_coord]

    def rotate_image(image, points,
                     alpha_range=15, n_turns=6, mode='edge'):
        nonlocal csv_points
        for alpha in np.linspace(-alpha_range, alpha_range, n_turns):
            alpha_rad = np.deg2rad(alpha)
            rotation_matrix = np.array([[cos(alpha_rad), -sin(alpha_rad)],
                                        [sin(alpha_rad), cos(alpha_rad)]])
            center = np.array([image.shape[0] // 2, image.shape[1] // 2])
            temp = (points - np.tile(center, NUMBER_OF_FACE_POINTS)).reshape(NUMBER_OF_FACE_POINTS, 2)
            rot_points = temp.dot(rotation_matrix).ravel() + np.tile(center, NUMBER_OF_FACE_POINTS)
            csv_points[get_current_file_name()] = rot_points
            save_image(rotate(image, angle=alpha, mode=mode))
            # show_points(rotate(image, angle=alpha, mode=mode), rot_points)

    print('[MINDATA] Loading ...')
    start_time = time()
    image_collection, face_points = load_data()
    print('time:', time() - start_time)
    # time: 112.18886137008667

    if shuffle_image:
        print('[MINDATA] Shuffle ...')
        start_time = time()
        image_collection, face_points = shuffle(image_collection, face_points)
        print('time:', time() - start_time)
        # time: 0.1250014305114746

    print('[MINDATA] Saving ...')
    start_time = time()
    np.save(join(dirname, 'dataset6_shuffle_grayscale.npy'), image_collection)
    np.save(join(dirname, 'labels6_shuffle.npy'), face_points)
    print('time:', time() - start_time)
    # time: 1.5156610012054443

    print('[DATA] Augmentation...')
    start_time = time()
    for image, points in zip(image_collection, face_points):
        csv_points[get_current_file_name()] = points
        save_image(image)  # original
        # show_points(image, points)
        if rotates:
            rotate_image(image, points)
        if flips:  # vertical flip
            points = flip_coord(image.shape, points)
            csv_points[get_current_file_name()] = points
            image = np.fliplr(image)
            save_image(image)
            # show_points(image, points)
            if rotates:  # rotate vertical flip
                rotate_image(image, points)
    print('time:', time() - start_time)
    # time: 333.11680722236633

    print('[CSV] Saving ...')
    start_time = time()
    save_csv(csv_points, join(dirname, 'gt.csv'))
    print('time:', time() - start_time)
    # time: 4.312551975250244


if __name__ == '__main__':
    gt_dir = 'tests\\00_test_img_input\\train\\gt.csv'
    train_dir = 'tests\\00_test_img_input\\train\\images'
    save_dir = 'tests\\00_test_img_input\\data84_shuffle'
    img_dir = join(save_dir, 'images')

    # gt_coord = read_csv(gt_dir)
    # data_augmentation(gt_coord, train_dir, save_dir)
