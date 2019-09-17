import cv2
import os
import sys
import numpy as np

OUTPUTS = './Outputs'
PATH = './Hands'
FEATURES_FILE = os.path.join(OUTPUTS,'sift_features.txt')
DESC_FILE = os.path.join(OUTPUTS,'sift_descriptors.txt')
ORDER_FILE = os.path.join(OUTPUTS,'sift_order.txt')
NFEATURES = 100
files = list()

try:
    os.remove(FEATURES_FILE)
except: pass
try:
    os.remove(DESC_FILE)
except: pass


def sifting(file):
    img_gray = cv2.imread(os.path.join(PATH, file), 0)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURES)
    kps, desc = sift.detectAndCompute(img_gray, None)    # print(len(kps))
    # print(desc[0], len(desc[0]))
    ps, temp = [], []
    for point in kps:
        ps.append( str(point.pt[0]) + ',' + str(point.pt[1]) + ',' + str(point.size) + ',' + str(point.angle) + ',' + str(
            point.response) + ',' + str(point.octave) + ',' + str(point.class_id) + ';' )

    f = open(FEATURES_FILE, 'a')
    for point in ps:
        f.write(point)
    f.write('\n')
    f.close()

    f = open(DESC_FILE, 'a')
    for item in desc:
        one = ''
        for i in item:
            one += (str(i)+',')
        f.write("%s," % one[:-1])
    f.write('\n')
    f.close()
    # for d in desc:
    #     temp.append(d)
    # with open(DESC_FILE, 'w') as f:
    #     f.writelines(["%s\n" % item for item in temp])
    # f = open(DESC_FILE, 'a')
    # for d in desc:
    #     temp.append(d)
    # f.write(temp)
    # f.write('\n')
    # f.close()

def main():
    if len(sys.argv) <= 1:
        for file in os.listdir(PATH):
            if file.endswith('.jpg'):
                files.append(file[5:-4])
                print(file)
                sifting(file)
    else:
        file = sys.argv[1]
        files.append(file)
        print(file)
        sifting(file)

    with open(ORDER_FILE, 'w') as f:
        for item in files:
            f.write("%s\n" % item)
    f.close()

if __name__ == '__main__':
    main()