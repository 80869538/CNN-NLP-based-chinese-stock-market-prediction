#!/usr/bin/python
import json
import csv
import re
import tushare as ts
import time
from config import *
# output file name: input/stockPrices_raw.json
# json structure: crawl daily price data from yahoo finance
#          ticker
#         /  |   \       
#     open close adjust ...
#       /    |     \
#    dates dates  dates ...

def get_stock_Prices():
    fin = csv.reader(open('./input/finished.reuters'))
    output = './input/stockPrices_raw.json'
    priceSet = {}
    priceSet['399300'] = repeatDownload('399300',index=True) # download S&P 500
    for num, line in enumerate(fin):
        print(line)
        code = line[0]
        repeattimes = 0
        while 1:
            repeattimes+=1
            try:
                if repeattimes>=3:
                    break
                priceSet[code] = repeatDownload(code)
                break
            except:
                continue
        # if num >= 0: break for testing purpose
    with open(output, 'w') as outfile:
        json.dump(priceSet, outfile, indent=4)

def repeatDownload(ticker,index=False):
    print("正在获取"+ticker)
    data = ts.get_h_data(ticker,start=START_DATE,end=END_DATE,index=index)
    data.to_csv(str('./input/CsvFileFor'+ticker+'.csv'))
    priceStr = PRICE(ticker)
    print(priceStr)
    time.sleep(30)
    return priceStr

def PRICE(ticker):
    file_name = './input/CsvFileFor'+ticker+'.csv'
    csv_file = csv.reader(open(file_name))
    # get historical price
    ticker_price = {}
    index = ['open', 'high', 'close', 'low','volume']
    for typeName in index:
        ticker_price[typeName] = {}
    for num, line in enumerate(csv_file):
        date = line[0]
        # check if the date type matched with the standard type
        if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date): continue
        # open, high, low, close, volume, adjClose : 1,2,3,4,5,6

        for num, typeName in enumerate(index):
            ticker_price[typeName][date] = round(float(line[num + 1]),2)

    return ticker_price


if __name__ == "__main__":
    get_stock_Prices()