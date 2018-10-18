from model_cnn import *
import matplotlib.pyplot as plt
clusters = 2
X_train, y_train, X_valid, y_valid, X_test, y_test = get_Feature_Label(clusters=clusters)
model = CNN(clusters)
# train_error = []
# valid_error = []
# test_error = []
#
# for nb_epoch in range(1,31):
#     temp_train = []
#     temp_valid = []
#     temp_test = []
#     for trail in range(1, 10):
#
#         train,valid,test = evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test,nb_epoch)
#         temp_train.append(train)
#         temp_valid.append(valid)
#         temp_test.append(test)
#     train_error.append(sum(temp_train)/len(temp_train))
#     valid_error.append(sum(temp_valid) / len(temp_valid))
#     test_error.append(sum(temp_test) / len(temp_test))
# plt.plot(train_error,label = "train_error")
# plt.plot(valid_error,label = "valid_error")
# plt.plot(test_error,label = "test_error")
# plt.show()
test_error = []

for thres in [float(x)/1000 for x in range(500,751,10)]:
    temp_train = []
    temp_valid = []
    temp_test = []
    for trail in range(1, 3):
        train,valid,test = evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test,thres=thres)
        temp_test.append(test)
    test_error.append(1-(sum(temp_test) / len(temp_test)))
plt.plot(test_error,label = "test_error")
plt.show()