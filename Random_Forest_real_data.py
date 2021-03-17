from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import csv

data_x = []
data_y = []
with open('../NLP/newlistfile.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        words = row[1][2:len(row[1])-2].replace("\"","").replace("\\","").replace("'", "").split(", ")
        data_x.append(' '.join(words))
        data_y.append(row[2])

# TF-IDF

tfidfv = TfidfVectorizer().fit(data_x)
transformed_data_x = tfidfv.transform(data_x)
# print(tfidfv.vocabulary_)
X_train, X_test, y_train, y_test = train_test_split(transformed_data_x, data_y, test_size=0.2) # train,test로 나눔



# RandomForest 분류 모델 생성
# n_estimators=10: 의사결정트리의 개수를 10으로 설정
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)


# 테스트셋을 이용해 target data 예측
prediction = rfc.predict(X_test)

print("정확도: ", accuracy_score(prediction, y_test))


print(classification_report(prediction, y_test))