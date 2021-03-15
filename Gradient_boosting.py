from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

# training data 설정(sklearn에서 제공하는 데이터셋 이용)
iris_dataset = load_iris()

# 150개 iris 데이터에서 120개를 훈련 set으로, 30개를 테스트set으로 사용
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2)


# 특성 스케일링: 각 특성의 범위가 크면 머신러닝 모델을 제대로 훈련시킬 수 없기 때문에 특성들을 스케일링해주어야 한다. -> 정규화 or 표준화
# 여기선 표준화 사용
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# 그래디언트부스팅 + 그리드서치 로 모델 학습
gb = GradientBoostingClassifier(random_state=1)
param_grid = [{'n_estimators': range(5, 50, 10), 'max_features': range(1, 4), 'max_depth': range(3, 5), 'learning_rate': np.linspace(0.1, 1, 10)}]
gs = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X_train_std, y_train)

# 그리드서치 학습 결과 출력
print('베스트 하이퍼 파라미터: {0}'.format(gs.best_params_))
print('베스트 하이퍼 파라미터 일 때 정확도: {0:.2f}'.format(gs.best_score_))


# 최적화 모델 추출
model = gs.best_estimator_

# 테스트셋 정확도 출력
prediction = model.predict(X_test_std)

print("정확도: ", accuracy_score(prediction, y_test))

print(classification_report(prediction, y_test))