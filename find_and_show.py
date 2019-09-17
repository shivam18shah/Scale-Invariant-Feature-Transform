import cv2
import os
import sys
import numpy as np

OUTPUTS = './Outputs'
PATH = './Hands'
files = []
FEATURES_FILE = os.path.join(OUTPUTS,'sift_features.txt')
ORDER_FILE = os.path.join(OUTPUTS,'sift_order.txt')
DESC_FILE = os.path.join(OUTPUTS,'sift_descriptors.txt')


def show_features(file):
    lines = [ line[:-1] for line in open(ORDER_FILE, 'r') ]
    # print('For File',file)
    try:
        number = lines.index(str(file))
    except:
        print('File not found')
        return
    # print(number)
    kps = []
    f = open(FEATURES_FILE, 'r')
    for i, line in enumerate(f):
        if i==number:
            keypoints = line
    f.close()

    for line in keypoints.split(';')[:-1]:
        kp = str(line).split(',')
        kp = cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), _size=float(kp[2]), _angle=float(kp[3]), _response=float(kp[4]), _octave=int(kp[5]), _class_id=int(kp[6]))
        kps.append(kp)
    # print(len(kps))

    descriptors = []
    f = open(DESC_FILE, 'r')
    for i, line in enumerate(f):
        if i==number:
            descriptors = list(line.split(';')[:-1])
    # print(descriptors[0], len(descriptors))
    f.close()
    desc = []
    for d in descriptors:
        desc.append(d.split(','))

    # print('Descriptors:',len(desc), 'Each Descriptor:',len(desc[0]))


    img = cv2.imread(os.path.join(PATH, 'Hand_' + file + '.jpg'), 1)
    img = cv2.drawKeypoints(img, kps, outImage=np.array([]))

    cv2.imshow(file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    file = sys.argv[1]
    if file in ['n', None, '']:
        return
    show_features(file)


if __name__ == '__main__':
    main()