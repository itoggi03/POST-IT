from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# training data 설정(sklearn에서 제공하는 데이터셋 이용)
iris_dataset = load_iris()

# 150개 iris 데이터에서 120개를 훈련 set으로, 30개를 테스트set으로 사용
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2)

# RandomForest 분류 모델 생성
# n_estimators=10: 의사결정트리의 개수를 10으로 설정
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)


# 테스트셋을 이용해 target data 예측
prediction = rfc.predict(X_test)

print("정확도: ", accuracy_score(prediction, y_test))
print(classification_report(prediction, y_test))