#!/usr/bin/python
import json
import jieba
import datetime
import numpy as np
from config import *

def dateGenerator(numdays): # generate N days until now, eg [20151231, 20151230]
    base = datetime.datetime.strptime(str(END_DATE),"%Y-%m-%d")
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
    return set(date_list)


def readGlove(we_file, w2i_file, concat=True):
    npz = np.load(we_file)
    W1 = npz['arr_0'] #2001行，50列，每行代表一个关键词的词向量
    W2 = npz['arr_1'] #将emdedding生成的组词向量取出来
    with open(w2i_file) as f:
        word2idx = json.load(f)  #将所有关键词取出来,共2001个

    V = len(word2idx)
    if concat:
        We = np.hstack([W1, W2.T]) #都变成(V,D)形式再拼接到一起,变成2001行100列
        print("We.shape:", We.shape)
        print(V == We.shape[0])
    else:
        We = (W1 + W2.T) / 2
    return We

def padding(sentencesVec, keepNum):
    shape = sentencesVec.shape[0]
    ownLen = sentencesVec.shape[1]
    if ownLen < keepNum:
        return np.hstack((np.zeros([shape, keepNum-ownLen]), sentencesVec)).flatten('F') #小于保留单词数时前面用0补齐，是不是应该用flatten('F')
    else:
        return sentencesVec[:, -keepNum:].flatten('F') #大于时截断

def gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words=60, mtype="test"):
    # step 2, build feature matrix for training data
    loc = ''
    input_files = ['./input/news.csv']
    current_idx = 2
    dp = {} # only consider one news for a company everyday
    cnt = 0
    testDates = dateGenerator(100) #生成截止到目前的前100天的数列
    shape = wordEmbedding.shape[1] #获取到每一个词向量的维数
    features = np.zeros([0, max_words * shape])      #最多取max_words个词向量做特征?,横着的，一行代表一条新闻
    labels = []
    for file in input_files:
        for line in open(loc + file): #遍历获取新闻信息的每一行
            line = line.strip().split(',')
            if len(line) != 5 :
                continue
            ticker,headline,day= line[0],line[1],line[3]
            date = (datetime.datetime.strptime(day,'%Y-%m-%d')+datetime.timedelta(days=3)).strftime('%Y%m%d')
            while date not in priceDt['short'][ticker]:
                if date>END_DATE:
                    date = datetime.datetime.strptime(END_DATE, '%Y-%m-%d').strftime('%Y%m%d')
                    break
                date = (datetime.datetime.strptime(date, '%Y%m%d') + datetime.timedelta(days=1)).strftime('%Y%m%d')
            if date not in priceDt['short'][ticker]:
                continue
            if cnt >= 5000:
                break
            cnt += 1
            print(cnt)
            if mtype == "test" and date not in testDates: continue #跳过不属于测试集的信息
            if mtype == "train" and date in testDates: continue #如果属于测试集但type是训练 则跳过该信息
            # 2.1 tokenize sentense, check if the word belongs to the top words, unify the format of words
            tokens = jieba.cut(headline) #将headline和body字符串转化为单词
            #tokens = [t for t in tokens if t in stopWords]
            #tokens = [t for t in tokens if t in topWords]
            # 2.2 create word2idx/idx2word list, and a list to count the occurence of words
            sentencesVec = np.zeros([shape, 0]) #
            for t in tokens: #对一条新闻中的标题与新闻中的每个单词找到其对应的特征向量，形成一个行向量
                if t not in word2idx:
                    continue #不属于关键词,unknown属于关键词
                sentencesVec = np.hstack((sentencesVec, np.matrix(wordEmbedding[word2idx[t]]).T)) #wordEmbedding是读取出来的词向量矩阵，从2001行中取出相应的行转为列向量，一个句子占100行
            features = np.vstack((features, padding(sentencesVec, max_words)))#将每条新闻占100行20列，其中每条新闻最多单词量不超过max——words，小于则在前面用0补全。为何feature matrix不是100的倍数啊啊啊啊
            labels.append(round(priceDt['short'][ticker][date], 6)) #将该股当日价格保存到lable 应给有三种呀
    features = np.array(features)
    labels = np.matrix(labels)
    featureMatrix = np.concatenate((features, labels.T), axis=1) #labels.T是将labels转为列向量，和每条新闻相连
    fileName = './input/featureMatrix_' + mtype + '_short.csv'
    np.savetxt(fileName, featureMatrix, fmt="%s")

def build(wordEmbedding, w2i_file, max_words=60):
    with open('./input/stockReturns.json') as data_file:
        priceDt = json.load(data_file)
    with open(w2i_file) as data_file:    
        word2idx = json.load(data_file)
    
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "train") #除了找到的前一白天为测试集 其他均为训练集
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "test")
    
                
                    
def main(we, w2i_file):
    wordEmbedding = readGlove(we, w2i_file)
    build(wordEmbedding, w2i_file, MAX_WORDS)


if __name__ == "__main__":
    we = './input/glove_model_50.npz'
    w2i_file = "./input/word2idx.json"
    main(we, w2i_file)