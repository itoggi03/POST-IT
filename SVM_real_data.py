from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 샘플 데이터(사이킷런에서 제공하는 붓꽃품종분류 데이터셋)
iris_dataset = datasets.load_iris()

# print(iris['data'])
# print(iris['target'])

# 샘플 데이터를 훈련셋과 테스트셋으로 분리(150개의 샘플 중 120개(80%)를 훈련셋으로, 30개(20%)를 테스트셋으로 사용)
# X_train: 훈련셋의 특성들이 담겨 있음. 행은 샘플, 열은 특성(4가지 특성)을 나타냄
# y_train: 훈련셋의 타겟들이 담겨 있음. 0/1/2가 각각 품종을 의미
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2)

# print(X_train)
# print(y_train)

# 특성 스케일링: 각 특성의 범위가 크면 머신러닝 모델을 제대로 훈련시킬 수 없기 때문에 특성들을 스케일링해주어야 한다. -> 정규화 or 표준화
# 여기선 표준화 사용
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# SVM 훈련
svm_model = svm.SVC(kernel='rbf', C=8, gamma=0.1)

svm_model.fit(X_train, y_train)

# 테스트셋으로 성능 확인
y_predict = svm_model.predict(X_test_std)

# 정확도 출력
print("prediction accuracy: {:.2f}".format(np.mean(y_predict == y_test)))