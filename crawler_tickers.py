import tushare as ts
data = ts.get_stock_basics()
data.to_csv('./input/tiker_list.csv',columns=['name'])