import cv2
import os
import sys
import numpy as np

epsilon = 0.0000001
PATH = './Hands'
TEST = './test'
OUTPUTS = './Outputs'
ORDER_FILE = os.path.join(OUTPUTS,'sift_order.txt')
DESC_FILE = os.path.join(OUTPUTS,'sift_descriptors.txt')
FEATURES_FILE = os.path.join(OUTPUTS,'sift_features.txt')
NFEATURES = 100
# test_file = input('Enter file id to rank similar images ')

def test_to_array(file):
    filename = os.path.join(TEST,'Hand_'+file+'.jpg')
    # print(file)

    img_gray = cv2.imread(filename, 0)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURES)
    kps, desc = sift.detectAndCompute(img_gray, None)
    ps = list()
    for point in kps:
        for attr in [ point.pt[0] , point.pt[1] , point.size , point.angle , point.response , point.octave , point.class_id ]:
            ps.append(attr)
    # print(type(ps))
    # print(type(desc[0][0]))
    for item in desc:
        for i in item:
            ps.append(i)
    return ps


def bring_kps():
    lines = [line[:-1] for line in open(FEATURES_FILE, 'r')]
    # print(len(lines)) # size of training set
    kps = []
    for line in lines:
        kp = []
        for phrase in line.split(';')[:-1]:
            ones = []
            for word in phrase.split(','):
                ones.append(float(word))
            kp.append(ones)
        kps.append(kp)

    return kps

def bring_desc():
    lines = [line[:-1] for line in open(DESC_FILE, 'r')]
    descs = []
    for line in lines:
        # print('Line:',type(line), len(line))
        descs.append(line.split(',')[:-1])

    # descs is a list of 33 with a list of 12800 in each element
    return descs


def bring_all():
    kps = bring_kps()
    descs = bring_desc()
    # print(len(kps), len(descs))
    # print(len(kps[0]), len(descs[0]))
    all_set = []
    for i in range(len(kps)):
        one_set = []
        # appending the list of keypoints
        for j in range(len(kps[i])):
            for k in kps[i][j]:
                one_set.append(float(k))
        # appending the list of descriptors
        for j in range(len(descs[0])):
            one_set.append(float(descs[0][j]))
        # appending the whole list to the comparing set
        all_set.append(one_set)
    # print('All set: ',len(all_set[0]))
    return all_set

def compare(data, test):
    # print(len(data), len(test))
    res = []
    for i in range(len(data)):
        temp = 0
        for j in range(len(data[0])):
            temp += (test[j] - data[i][j])**2
        temp = temp**.5
        res.append(temp)
    res /= (sum(res)+epsilon)
    return res

def get_rank(res2):
    lines = [line[:-1] for line in open(ORDER_FILE, 'r')]
    for i in range(len(lines)):
        res2[i] = lines[res2[i]]
    # print(res2)
    return res2

def main():
    test_desc = test_to_array(sys.argv[1])

    # print(len(test_desc))
    all_set = bring_all()
    # print(len(all_set), len(all_set[0]))
    # print(len(desc[0]))
    # print('Descriptors:', len(desc), 'Each Descriptor:', len(desc[0]))
    res = compare(all_set, test_desc)
    # print(res)
    # sort array indices asending
    dummy = np.sort(res)
    try:
        number_of_results = min(len(res), int(sys.argv[2]))
    except:
        number_of_results = len(res)
    res2 = [i[0] for i in sorted(enumerate(res), key=lambda x:x[1])]
    res2 = get_rank(res2)
    for i in range(len(dummy)):
        x = np.nonzero(res==dummy[i])[0][0]
        dummy[i] = res[x]
    for i in range(number_of_results):
        print(i+1, res2[i], 1-dummy[i], '\n')


if __name__ == '__main__':
    main()