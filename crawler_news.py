import csv
import tushare as ts
import numpy as np
import datetime
from config import *
def get_news(start_date,end_date):
    data = csv.reader(open('./input/tiker_list.csv','r'))
    news = np.zeros([0,5])
    start_date = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    codes = []
    for num,line in enumerate(data):
        if num!=0 and num<=NUM_DATA: #for test
            print("开始搜索有关" + line[1] + "(" + line[0] + ')' + "的新闻...")
            codes.append(line[0])
            day = start_date
            while day <= end_date:
                try:
                    hight = np.array(ts.get_notices(line[0], day.strftime('%Y-%m-%d'))).shape[0]
                    news = np.vstack((news,np.hstack((np.array([[line[0]]*hight]).T,ts.get_notices(line[0], day.strftime('%Y-%m-%d'))))))
                    print("获取在" + day.strftime('%Y-%m-%d') + '的新闻成功')
                except IndexError:
                    pass
                except:
                    print("获取在" + day.strftime('%Y-%m-%d') + '的新闻失败，正在重试')
                    continue
                day+=datetime.timedelta(days=1)
    writer = csv.writer(open('./input/news.csv','w'))
    writer.writerows(news)
    codes = np.array([list(codes)])
    codes = codes.T
    out = open('./input/finished.reuters','w',newline='')
    csv_write = csv.writer(out)
    csv_write.writerows(codes)

def main():
    get_news(START_DATE,END_DATE)
if __name__ == "__main__":
    main()