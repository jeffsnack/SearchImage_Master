# -*- coding: utf-8 -*-
import keras
import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from keras.applications.vgg16 import preprocess_input


class VGGNet:
    def __init__(self):
        keras.backend.clear_session()
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        #加载自己训练的模型
        self.model.load_weights(r'D:\python_projects\SearchImage-master\model_pro\model\vgg16_use7.h5',by_name=True)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        #归一化
        norm_feat = feat[0]/LA.norm(feat[0])
        # print(feat.shape)
        return norm_feat

if __name__ == '__main__':
    #url =r'D:\python_projects\dogs-vs-cats-redux-kernels-edition\train\cat.1.jpg'
    vgg=VGGNet()
    #result =vgg.extract_feat(url)
    #print(result)