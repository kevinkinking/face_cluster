import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import cv2
import caffe
import scipy
import config

def init_model(prototxt, caffemodel):
    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
        return False
    if not os.path.isfile(prototxt):
        print ("prototxt not found!")
        return False
    caffe.set_mode_cpu()
    global net
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    if net != None:
        return True
    else:
        print ("load model is failed, but prototxt and caffemodel is exit!")
        return False

def feature_extract(img):
    if img.shape[-1] == 3:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[-1] == 4:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        print 'input image channels must in (3, 4)'
        return
    if input_img.shape[0] != config.face_img_height or input_img.shape[1] != config.face_img_height:
        input_img = cv2.resize(input_img, (config.face_img_width, config.face_img_height), interpolation = cv2.INTER_CUBIC)
    img_blobinp = input_img[np.newaxis, np.newaxis, :, :] / 255.0
    net.blobs['data'].reshape(*img_blobinp.shape)
    net.blobs['data'].data[...] = img_blobinp
    net.blobs['data'].data.shape
    net.forward()
    return net.blobs['eltwise_fc1'].data[0].copy()#must use copy,data is like c++ pointer

def feature_similar(feature_to_compare, feature):
    #error input then return 0
    if len(feature_to_compare) == 0 or len(feature) == 0:
        return 0
    similar = 1 - scipy.spatial.distance.cosine(feature_to_compare, feature)
    return similar



def feature_similar_s(feature_to_compare, features_back):
    if len(features_back) == 0:  
        return []
    similars = []
    for feature in features_back:
        similar = 1 - scipy.spatial.distance.cosine(feature, feature_to_compare)
        similars.append(similar)
    return similars

if __name__ == '__main__':
    init_model('./face_feature/models/LCNN_deploy.prototxt', './face_feature/models/LCNN_iter_3560000.caffemodel')
    img_left = cv2.imread('./face_feature/imgs_test/1.jpg')
    img_right = cv2.imread('./face_feature/imgs_test/4.jpg')
    feature_left = feature_extract(img_left)
    feature_right = feature_extract(img_right)
    print feature_similar(feature_left, feature_right)