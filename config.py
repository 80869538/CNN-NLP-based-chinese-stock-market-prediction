# 1. 直接运行crawler_tickers.py获取沪深股票列表
# 2. 运行crawler_news.py 与 crawler_yahoo_finance 并调整参数
##crawler_news.py and crawler_yahoo_finance.py configuration
START_DATE = '2014-01-01'
END_DATE = '2016-01-01'
NUM_DATA = 150 #需要测试的股票个数，最高为3505，及全部沪深股票
# 3. 直接运行create_label.py 计算股票相对回报率,基准为沪深300指数
# 4. 运行get_glove通过语料库计算生成词向量矩阵及关键词列表
##get_glove configuration
NUM_EPOCH = 1500 #1500次训练过后取向收敛
V_D = 50 #词向量维数
NUM_KEYS = 3000 #关键词个数
# 5. 运行genFeatureMatrix 为搜集到的新闻材料生成特征矩阵
##genFeatureMatrix configuration
NUM_NEWS = 800 #总训练新闻条数，上限为news.csv中总新闻条数
MAX_WORDS = 25 #每条新闻最大词语量
# 6.运行model_cnn.py
NUM_CON = 3 #卷积核维数
# 7.get answers,label0 代表跌，label1代表涨