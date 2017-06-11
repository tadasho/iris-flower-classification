from sklearn import svm, metrics
import random, re

# アヤメのデータを読み込む
# アヤメデータ https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv
csv = []
with open ('iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        line = line.strip()
        cols = line.split(',')

        fn = lambda n :float(n) if re.match(r'^[0-9\.]+$', n) else n
        csv.append(cols)

# ヘッダーの削除
del csv[0]

# シャッフル
random.shuffle(csv)

# 学習用とテスト用に分ける
# 前から2/3が学習用、残り1/3がテスト用
total_len = len(csv)
train_len = int(total_len * 2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(total_len):
    # 配列の1番目から5番目までを取得 -> [4]を含まない
    data = csv[i][0:4]

    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)
    
# データの学習
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 正解率を導出
ac_score = metrics.accuracy_score(test_label, pre)
print("正解率 = ", ac_score)
