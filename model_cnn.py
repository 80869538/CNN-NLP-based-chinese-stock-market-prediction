#!/usr/bin/python
import random
import numpy as np
import operator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical  #
from sklearn.metrics import confusion_matrix
from config import *
import csv
def value2int(y, clusters=2):
    label = np.copy(y)
    label[y < np.percentile(y, 100 / clusters)] = 0
    for i in range(1, clusters):
        label[y > np.percentile(y, 100 * i / clusters)] = i
    return label

def value2int_simple(y): #股价小于0的标记为0，否则标记为1
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label

def get_Feature_Label(clusters=2, hasJunk=True):
    data = np.genfromtxt('./input/featureMatrix_train_short.csv')
    test = np.genfromtxt('./input/featureMatrix_test_short.csv')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1] #X为词向量矩阵，每100行代表一条新闻，每一列地表一个单词
    print("Positive News Ratio", sum(y > 0) * 1. / (sum(y > 0) + sum(y < 0)))
    label = to_categorical(value2int_simple(y)).astype("int") # using direction to label  用涨跌来做label
    #label = to_categorical(value2int(y, clusters)).astype("int") # using quantile to label
    validation_ratio = 0.2
    X = X.reshape(X.shape[0],MAX_WORDS, V_D*2, 1).astype('float32')  #每行代表一个句子，每个句子有20个词，每个词向量维数是50
    #这样就变成了X.shape[0]个句子，其中每个句子有20行，每行代表一个单词，每个单词占50列
    print(data.shape[0])
    D = int(data.shape[0] * validation_ratio)  # total number of validation data
    X_train, y_train, X_valid, y_valid = X[:-D], label[:-D,:], X[-D:], label[-D:,:]
    # file = open('./input/train_data.csv','w')
    # writer = csv.writer(file)
    # writer.writerows(y_train)
    # writer.writerows(y_valid)
    X_test, y_test = test[:, :-1], test[:, -1]
    # writer.writerows(y_test)

    print("Positive News Ratio", sum(y_test > 0) * 1. / (sum(y_test > 0) + sum(y_test < 0)))
    X_test = X_test.reshape(X_test.shape[0],MAX_WORDS, V_D*2, 1).astype('float32')
    y_test = to_categorical(value2int_simple(y_test)).astype("int")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def CNN(clusters):
    model = Sequential()
    model.add(Convolution2D(64, NUM_CON, V_D*2, border_mode='valid', input_shape=(MAX_WORDS, V_D*2, 1), activation='relu')) #一个filter 扫三个词
    model.add(MaxPooling2D(pool_size=(MAX_WORDS-NUM_CON+1 , 1))) #将每个filter的所有输出结果转换成 （18，1）的列向量，方法为取每个结果的最大值
    model.add(Dropout(0.4))
    model.add(Flatten())#把多维压成一维
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(clusters, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test,nb_epoch = 13,thres = 0.5):
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=nb_epoch, batch_size=1024, verbose=2)
    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_train), axis=-1)
    conf = confusion_matrix(np.argmax(y_train, axis=-1), predictions)
    train_error = (conf[0, 0] + conf[1, 1])/(conf[0, 0] + conf[1, 1] + conf[1, 0] + conf[0, 1])
    print(train_error)
    predictions = np.argmax(model.predict(X_valid), axis=-1) #predict返回的是取每个label的概率大小，最后应选概率大的
    conf = confusion_matrix(np.argmax(y_valid, axis=-1), predictions)
    print(conf)
    valid_error = (conf[0, 0] + conf[1, 1])/(conf[0, 0] + conf[1, 1] + conf[1, 0] + conf[0, 1])
    print('Training Set Accuracy:'+str(valid_error))
    for i in range(clusters):
        print("Valid Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))
    # calculate predictions
    predictions =model.predict(X_test)
    thres = thres; y_cut = (predictions[:,0] > thres) | (predictions[:,1] > thres) # cut y value and leave the better result
    predictions = np.argmax(predictions[y_cut], axis=-1)
    conf = confusion_matrix(np.argmax(y_test[y_cut], axis=-1), predictions)
    print("Test on %d samples" % (len(y_test[y_cut])))
    print(conf)
    print(conf[0,0])
    test_error = (conf[0, 0] + conf[1, 1])/(conf[0, 0] + conf[1, 1] + conf[1, 0] + conf[0, 1])
    print('Test Set Accuracy:'+str(test_error))
    for i in range(clusters):
        print("Test Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))
    return train_error,valid_error,test_error


def model_selection(clusters): # random sampling is better than grid search
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_Feature_Label(clusters=clusters)

    for i in range(30):
        print("Trial:", i)
        model = CNN(clusters)
        evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test)
        

def main():
    clusters = 2
    model_selection(clusters)
    


if __name__ == "__main__":
    main()
