import csv
news = csv.reader(open('./input/news.csv'))
headline = []
cnt = 0
for line in news:
    cnt+=1
    headline.append([line[1]])
print(cnt)
writer = csv.writer(open('./input/corpus.csv','w'))
writer.writerows(headline)