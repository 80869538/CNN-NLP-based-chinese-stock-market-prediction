#!/usr/bin/python
import json
import datetime
from math import log

# input file name: ./input/stockPrices_raw.json
# output file name: ./input/stockReturns.json
# json structure: crawl daily price data from yahoo finance
#          term (short/mid/long)
#         /         |         \
#   ticker A   ticker B   ticker C
#      /   \      /   \      /   \
#  date1 date2 date1 date2 date1 date2
# 
# Note: short: 1 day return, mid: 7 day return, long 28 day return


# calc long/mid term influence
def calc_mid_long_return(ticker, date, delta, priceSet):
    #字符串转化为日期
    baseDate = datetime.datetime.strptime(date, "%Y-%m-%d")
    #相隔一天的时间差和相隔delta的时间差
    prevDate = (baseDate - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    nextDate = (baseDate + datetime.timedelta(days=delta)).strftime("%Y-%m-%d")

    try:
        #对数形式计算相对回报率
        return_self = log(priceSet[ticker]['close'][nextDate]) - log(priceSet[ticker]['close'][prevDate])
        return_sp500 = log(priceSet['399300']['close'][nextDate]) - log(priceSet['399300']['close'][prevDate])
        return True, round(return_self - return_sp500, 4) # relative return
    except:
        return False, 0

def main():
    raw_price_file = './input/stockPrices_raw.json'
    with open(raw_price_file) as file:
        print("Loading price info ...")
        priceSet = json.load(file)
        dateSet = priceSet['399300']['close'].keys()

    returns = {'short': {}, 'mid': {}, 'long': {}} # 1-depth dictionary
    for ticker in priceSet:
        if ticker != '399300':
            for term in ['short', 'mid', 'long']:
                returns[term][ticker] = {} # 2-depth dictionary
            for day in dateSet:
                date = datetime.datetime.strptime(day, "%Y-%m-%d").strftime("%Y%m%d") # change date 2014-01-01 to 20140101
                tag_short, return_short = calc_mid_long_return(ticker, day, 0, priceSet)
                tag_mid, return_mid = calc_mid_long_return(ticker, day, 6, priceSet)
                tag_long, return_long = calc_mid_long_return(ticker, day, 27, priceSet)
                if tag_short: returns['short'][ticker][date] = return_short
                if tag_mid: returns['mid'][ticker][date] = return_mid
                if tag_long: returns['long'][ticker][date] = return_long

    with open('./input/stockReturns.json', 'w') as outfile:
        json.dump(returns, outfile, indent=4)

if __name__ == "__main__":
    main()