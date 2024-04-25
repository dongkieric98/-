# 전력사용량 예측 공모전
## 1. 공모전 개요
#### [배경] 
- 안정적이고 효율적인 에너지 공급을 위해서는 전력 사용량에 대한 정확한 예측이 필요합니다.
- 따라서 한국에너지공단에서는 전력 사용량 예측 시뮬레이션을 통한 효율적인 인공지능 알고리즘 발굴을 목표로 본 대회를 개최

#### [설명]
- 건물 정보와 시공간 정보를 활용하여 특정 시점의 전력 사용량을 예측하는 AI 모델 개발
- train 데이터셋을 바탕으로 test 데이터셋의 전력사용량을 예측하여 SMAPE값이 낮게 나오도록 전력사용량 예측 모델을 개발

#### [주최/주관]
- 주최: 한국 에너지공단
- 주관: 데이콘(Dacon)

#### [리더보드]
- 심사기준: SMAPE(Symmetric Mean Absolute Percentage Error)
- Public Score: 2022.08.25 ~ 2022.08.27의 실제 전력사용량 데이터로 측정
- Private Score: 2022.08.25 ~ 2022.08.31의 실제 전력사용량 데이터로 측정

#### [평가]
- 1차 평가: 리더보드 Private Score
- 2차 평가: Private Score 상위 10팀 코드 및 PPT 제출 후 코드 평가

<br/></br>

## 2. 개발환경 (Environment)
- 운영체제: Window 11 Home
- 개발언어: Python 3.9.5
- 개발도구: Visual Studio Code (VSCode)

<br/></br>

## 3. 라이브러리 (Library)
- numpy == 1.26.4
- pandas == 2.2.2
- seaborn == 0.13.2
- xgboost == 2.0.3
- scikit-learn == 1.4.2

<br/></br>

## 4. 파일 구조
- 'data/' : 이 폴더에는 건물정보데이터, 학습데이터, 테스트데이터, 답안제출양식데이터, 최종제출답안데이터가 있습니다.
  - 'building_info.csv' : 건물정보데이터로 csv 형식
  - 'train.csv' : 모델 학습데이터(2022.06.01 ~ 2022.08.24)
  - 'test.csv' : 모델 테스트데이터(2022.08.25 ~ 2022.08.31)
  - 'submission.csv' : Dacon에 제출해야하는 답안 제출 양식
  - 'submission_final.csv' : 최종 제출 답안
- 'electricity.ipynb' : 전력사용량 예측 main 코드
- 'electricity_EDA' : 전력사용량 EDA/시각화 코드
- 'electricity.pdf' : 전력사용량 공모전 PDF 파일
  
<br/></br>

## 5. 대회 진행시 고민 & 어려웠던 점
#### [건물별 모델링 vs 건물유형별 모델링]
- 어려움: 건물별 모델링과 건물유형별 모델링 중 어떤 방식으로 모델링 해야할 지 고민함
- 해결방안: 건물별 모델링을 할 때, 학습가능한 데이터의 개수가 적고, 새로운 건물이 들어왔을 때, 해당 건물에 대한 예측이 떨어지기 때문에 건물유형별 모델링을 하기로 결정

#### [Validation Score와 PLB Score 간의 많은 차이]
- 문제점: SMAPE 값이 Validation Score에 비해 PLB Score가 훨씬 높게 나오는 문제 발생
- 해결방안: Train 데이터를 학습하는 과정에서 Overfitting이 발생하여 해당 문제가 발생하는 것으로 간주하여 모델링 할 때, Overfitting 방지에 적합한 모델을 사용 (XGBOOST, RANDOM FOREST)
- 실제결과: 해당 모델을 사용하여 Validation Score와 PLB Score의 차이를 줄일 수는 있었으나 여전히 높은 차이가 발생하여 문제를 완벽하게 해결하지는 못하였음

#### [Hyperparameter Tuning에 많은 시간 소모]
- 문제점: 컴퓨터의 사양이 좋지 않아 Grid Search를 사용하여 Hyperparameter Tuning을 하는데 너무 오랜 시간이 걸려 한번 모델을 돌리는데 수시간이 걸려 프로젝트를 진행하는데 차질이 생김
- 해결방안:
  - Grid Search를 사용하여 Hyperparameter Tuning을 할 때, Hyperparameter의 범위를 좁게 설정하여 Tuning을 진행
  - Random Search를 사용하여 Hyperparameter Tuning에 걸리는 시간을 보다 줄임
 
#### [Hyperparameter Tuning의 필요성]
- 문제점: 긴 시간에 걸쳐 몇번의 Hyperparameter Tuning을 시도하였으나 기본 설정한 Hyperparameter에 비해 모델의 성능이 더 떨어지는 현상 발생
- 해결방안: 기본 Hyperparameter를 사용하여 Tuning을 하고, Feature Engineering과 같은 다른 요소를 통해 모델의 성능을 향상 방안을 발견

#### [Feature Selection의 기준 잡기]
- 문제점: Target값과의 낮은 상관관계를 가진 Feature를 제거하고 모델을 학습시켰을 때, 보다 높은 성능을 가지고 있을 것이라고 예상하였으나 실제로 상관계수가 낮은 Feature를 포함한 모든 Feature를 넣었을 때, 더 좋은 성능이 나타나는 문제가 발생하여 무엇을 기준으로 Feature Selection을 해야하는지에 대한 문제가 발생
- 해결방안: 각각의 Feature를 직접 넣고 빼는 과정을 반복하여 가장 높은 Validation Score 값이 나오는 Feature를 선택하여 Train 데이터를 학습시킴

#### [파생변수 생성]
- 문제점: 모델의 성능을 높이기 위해 파생변수를 생성하려고 하였으나 어떤 변수를 생성해야할지에 대한 고민
- 해결방안: 선행연구와 논문을 통해 전력사용량에 있어 생성할 수 있는 파생변수를 파악하고 현재 내가 가진 Feature로 제작할 수 있는 파생변수를 생성

  <br/></br>
  
## 6. 건물정보 데이터(building) 전처리
#### [건물정보 데이터 결측치 비율]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/dcf31e5f-ea74-40aa-8c6a-b3caebef44ea)

#### [태양광용량, ESS저장용량, PCS저장용량 전처리]
- 해당 결측치 값이 -로 되어있고 데이터 타입도 object여서 '-'값으 0으로 바꾸고 데이터 타입을 float64로 변경

```
# 태양광이 없는 곳이므로 0값처리
building['태양광용량(kW)'] = building['태양광용량(kW)'].replace('-', 0)
building['태양광용량(kW)'] = building['태양광용량(kW)'].astype('float64')

# 대부분 결측치이고, 해당 기능이 없는 것이므로 0값처리
building['ESS저장용량(kWh)'] = building['ESS저장용량(kWh)'].replace('-', 0)
building['ESS저장용량(kWh)'] = building['ESS저장용량(kWh)'].astype('float64')
building['PCS용량(kW)'] = building['PCS용량(kW)'].replace('-', 0)
building['PCS용량(kW)'] = building['PCS용량(kW)'].astype('float64')
```

#### [냉방면적 결측치 처리]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/f9db84d7-5449-49f8-9d37-c0d15daaa346)
- 냉방면적의 경우 결측치 값이 아파트에서 3개만 존재
- 해당 결측치를 아파트의 '연면적 : 냉방면적' (=1.27) 비율을 적용하여 냉방면적 결측치 처리
```
# '연면적 : 냉방면적 (=1.27)' 비율을 적용하여 냉방면적 결측치 처리
building['냉방면적(m2)'][64] = building['연면적(m2)'][64]/1.27
building['냉방면적(m2)'][65] = building['연면적(m2)'][65]/1.27
building['냉방면적(m2)'][67] = building['연면적(m2)'][67]/1.27
```

<br/></br>

## 7. 학습 데이터(train) 전처리

#### [파생변수 생성1 - 년/월/일/시/휴일]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/24a428ae-8388-4434-8a6d-07cfeddedf1c)
- 기존의 날짜 정보로 부족하다는 생각이 들어 년/월/일/시로 분리
- 평일과 휴일간의 전력사용량 차이가 클 것이라고 예상하여 휴일 변수 추가
```
# 평일/휴일 컬럼 만들기 - 평일과 휴일의 전력사용량 차이가 있음
train['일시'] = pd.to_datetime(train['일시'], format = '%Y%m%d %H')
def is_holiday(row):
    if row['일시'].date() in [pd.to_datetime('2022-06-06').date(), pd.to_datetime('2022-08-15').date()]:
        return 1
    elif row['일시'].dayofweek >= 5:  # 토요일(5) 또는 일요일(6)인 경우
        return 1
    else:
        return 0
train['휴일'] = train.apply(is_holiday, axis=1)

# 일시를 년, 월, 일 컬럼으로 분리(train) - 기존의 num_date_time을 사용하기 어려움
train['년'] = train['일시'].dt.year
train['월'] = train['일시'].dt.month
train['일'] = train['일시'].dt.day
train['시'] = train['일시'].dt.hour
```

#### [파생변수 생성2- sin time, cos time]
- 기존의 데이터로 시간의 주기성을 파악하기 어려움
- 시간의 주기성을 반영하는 sin time과 cos time을 추가
- building 데이터와 train 데이터를 '건물번호' 컬럼을 기준으로 병합
- 같은 연도이고 시간을 대체할 변수를 생성하였기에 년/일시 컬럼 삭제
```
# 시간 데이터 표준화 - 시간의 주기성을 반영하기 위해 사용
train['sin_time'] = np.sin(2*np.pi*train['시']/24)
train['cos_time'] = np.cos(2*np.pi*train['시']/24)

# 공통된 건물번호 컬럼을 기준으로 데이터 병합
train = pd.merge(building, train, on = '건물번호')
train.drop(['num_date_time', '일시', '년'], axis=1, inplace=True)
```

#### [날씨 데이터 결측치 처리]
- 기상청에서 날씨 데이터에 대해 결측치 처리하는 방식(interpolate)을 적용하여 결측치 처리
- Interpolate 함수 사용 근거: 날씨 데이터는 시간에 따라 연속적인 값을 가지는 특성이 있습니다. 예를 들어, 온도나 강수량 등의 데이터는 시간의 흐름에 따라 점진적으로 변화합니다. INTERPOLATE를 사용하면, 결측된 데이터 포인트를 인접한 데이터 포인트들을 기반으로 적절하게 추정함으로써 데이터의 연속성을 유지할 수 있습니다.
```
# 기상청에서 결측치 처리에 사용되는 interpolate를 이용해 결측치 처리
train['강수량(mm)'] = train['강수량(mm)'].interpolate(method = 'linear')
train['풍속(m/s)'] = train['풍속(m/s)'].interpolate(method = 'linear')
train['습도(%)'] = train['습도(%)'].interpolate(method = 'linear')
train['일조(hr)'] = train['일조(hr)'].interpolate(method = 'linear')
train['일사(MJ/m2)'] = train['일사(MJ/m2)'].interpolate(method = 'linear')
```

#### [파생변수 생성3 - 불쾌지수, 냉방효율, 체감온도]
- 인간의 체감정보와 에너지 효율과 관련된 파생변수 추가
- 전력 도메인에 따르면 불쾌지수와 체감온도의 경우 소비자 행동의 예측에 있어 중요한 지표이기에 사용
- 냉방효율의 경우 냉방 시스템의 효율성을 반영하고 평가하기 위해 사용
- 불쾌지수 = 1.8*기온 – 0.55*(1-습도/100) * (1.8*습도-26) + 32
- 체감온도 = 13.12 + 0.6125*기온 – 11.37*풍속𝟎.𝟏𝟔+ 0.3965*풍속𝟎.𝟏𝟔*기온
- 냉방효율 = 냉방면적 / 연면적 
```
# 파생변수 생성: 불쾌지수/냉방효율/체감온도
train['불쾌지수'] = 1.8 * train['기온(C)'] -0.55 * (1-train['습도(%)']/100) * (1.8 * train['습도(%)']-26) + 32
train['냉방효율'] = train['냉방면적(m2)'] / train['연면적(m2)']
train['체감온도'] = 13.12 + 0.6125 * train['기온(C)'] - 11.37*(train['풍속(m/s)']**0.16) + 0.3965 * (train['풍속(m/s)']**0.16)*train['기온(C)']
```

#### [태양광용량 전처리]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/a4f85a10-566f-4518-8234-d17616f2811d)
- 태양광용량의 경우 결측치가 64%에 해당
- 태양광용량의 결측치 비율이 0%와 100%인 건물유형이 존재하여 해당 건물유형은 모델링 과정에서 제거하고 진행
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/8dc5d820-9eb2-4c02-b16b-8045d2aa6063)
- 실제로 시각화를 해서 확인해본 결과 태양광용량의 유/무에 따른 전력사용량에서 유의미한 차이가 난다고 판단하여 태양광용량의 정도가 아닌 유/무로 변환하여 사용
```
# 태양광용량은 유무로 0과 1로 처리
train['태양광용량(kW)'] = train['태양광용량(kW)'].apply(lambda x : 1 if x != 0 else x)
train['태양광용량(kW)'] = train['태양광용량(kW)'].astype('int')
```

#### [불필요한 컬럼 삭제]
- 결측치 비율이 많고 전력사용량과의 상관관계도 적으며 활용할 방법이 없는 변수를 삭제
```
# 필요없거나 영향 없는 컬럼 삭제
train.drop(['ESS저장용량(kWh)', 'PCS용량(kW)', '강수량(mm)', '일조(hr)', '일사(MJ/m2)'], axis=1, inplace=True)
```
<br/></br>

## 8. 테스트 데이터(test) 전처리
- train 데이터와 동일한 방식으로 전처리
```
# 주말/공휴일 컬럼 만들기
test['일시'] = pd.to_datetime(test['일시'], format = '%Y%m%d %H')
def is_holiday(row):
    if row['일시'].date() in [pd.to_datetime('2022-06-06').date(), pd.to_datetime('2022-08-15').date()]:
        return 1
    elif row['일시'].dayofweek >= 5:  # 토요일(5) 또는 일요일(6)인 경우
        return 1
    else:
        return 0
test['휴일'] = test.apply(is_holiday, axis=1)

# 일시를 년, 월, 일 컬럼으로 분리(test)
test['년'] = test['일시'].dt.year
test['월'] = test['일시'].dt.month
test['일'] = test['일시'].dt.day
test['시'] = test['일시'].dt.hour

# test 시 컬럼 전처리
test['sin_time'] = np.sin(2*np.pi*test['시']/24)
test['cos_time'] = np.cos(2*np.pi*test['시']/24)

# 사용하지 않는 컬럼 삭제
test.drop(['num_date_time', '일시', '년'], axis=1, inplace=True)

# 공통된 건물번호 컬럼을 기준으로 데이터 병합
test = pd.merge(building, test, on = '건물번호')
```
```
# 건물유형 컬럼 처리 숫자로 변경
test['건물유형'] = test['건물유형'].replace('건물기타', 1)
test['건물유형'] = test['건물유형'].replace('공공', 2)
test['건물유형'] = test['건물유형'].replace('대학교', 3)
test['건물유형'] = test['건물유형'].replace('데이터센터', 4)
test['건물유형'] = test['건물유형'].replace('백화점및아울렛', 5)
test['건물유형'] = test['건물유형'].replace('병원', 6)
test['건물유형'] = test['건물유형'].replace('상용', 7)
test['건물유형'] = test['건물유형'].replace('아파트', 8)
test['건물유형'] = test['건물유형'].replace('연구소', 9)
test['건물유형'] = test['건물유형'].replace('지식산업센터', 10)
test['건물유형'] = test['건물유형'].replace('할인마트', 11)
test['건물유형'] = test['건물유형'].replace('호텔및리조트', 12)
```
```
# 파생변수 생성: 불쾌지수/냉방효율/체감온도
test['불쾌지수'] = 1.8 * test['기온(C)'] -0.55 * (1-test['습도(%)']/100) * (1.8 * test['습도(%)']-26) + 32
test['냉방효율'] = test['냉방면적(m2)'] / test['연면적(m2)']
test['체감온도'] = 13.12 + 0.6125 * test['기온(C)'] - 11.37*(test['풍속(m/s)']**0.16) + 0.3965 * (test['풍속(m/s)']**0.16) * test['기온(C)']

# 필요없거나 영향 없는 컬럼 삭제
test.drop(['ESS저장용량(kWh)', 'PCS용량(kW)', '강수량(mm)'], axis=1, inplace=True)
```

## 9. 평가지표 생성 - SMAPE

#### [SMAPE 공식]
- 공모전에서 평가 기준이 SMAPE 값이기에 해당 값을 생성

![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/8b278062-c250-4fb1-b291-8acf9dfcccfe)

- Ft(Forecast Value): t시점의 예측값
- At(Actual Value): t시점의 실제값

#### [SMAPE 특성]
- 대칭성: SMAPE는 실제 값이 예측 값보다 크든 작든 오차를 동일하게 취급함, 이것은 MAPE에서 나타나는 문제, 즉 과소 예측의 오차가 과대 예측보다 더 크게 나타나는 문제를 해결
- 제한된 범위: SMAPE의 값은 0%에서 200% 사이, 0%는 완벽한 예측을 나타내고, 200%는 예측이 매우 부정확할 때 나타나는 극단적인 경우를 나타냄

#### [SMAPE 주의점]
- 0 분모 문제: 실제 값과 예측 값이 모두 0인 경우, 분모가 0이 되어 계산할 수 없음, 이를 방지하기 위해 작은 상수 (epsilon)를 분모에 추가하기도 함
- 극단값: 예측이나 실제 값 중 하나가 극단적으로 클 때, SMAPE는 해당 값의 영향을 크게 받을 수 있음


## 10. 건물 유형별 데이터 모델링

#### [건물 유형별 데이터 모델링 근거]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/3564fb4d-fc79-427c-a440-a6b414050855)

1) 건물유형별 유형별과 건물번호별 모델링 중 어떤 방식으로 할까 고민하였으나 건물번호의 경우 번호별로 2040개의 데이터가 있고 건물유형은 16320개의 데이터가 존재
2) 모델링 과정에서 건물번호별로 모델링할 경우 데이터의 학습 개수가 적다고 판단하여 건물유형별 모델링을 적용
3) 건물유형별 전력소비량 평균값을 확인해본 결과 건물유형별로 유의미한 차이가 있다고 판단

#### [모델 선정 과정]
- 여러 모델을 사용해본 결과 Validation Score와 PLB Score 간의 차이가 많이 나는 현상을 발견
- SMAPE 값에 대해 Validation Score가 PLB Score에 비해 높게 나타나는 현상 발생
- 따라서 학습 데이터에 대해 Overfitting이 일어나 이를 주의하며 모델을 선정
- 이에 따라 사용해볼 모델이 XGBOOST와 RandomForest를 사용하기로 결정
- 각 모델 적용 후 결과를 확인하여 SMAPE값이 더 작게 나온 모델을 사용하기로 결정
  
#### [10-1. 건물기타 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/124ad416-4fb4-46a0-9142-f854c693a651)

- HeatMap을 통해 확인한 후 전력소비량과 상관관계가 높은 변수를 사용하기로 결정
- 사용변수: '태양광용량(kW)', '불쾌지수', '체감온도', '냉방효율', 'sin_time', 'cos_time', '휴일', '월', '일'
- XGBOOST와 RANDOMFOREST중 XGBOOST가 더 높은 성능을 보여 해당 모델 사용
- Parameter Tuning의 경우 아래 코드와 같이 진행
```
# 모델 종류 선정 및 하이퍼파라미터 설정
model = xgb.XGBRegressor(
     n_estimators = 500,
     learning_rate = 0.05,
     max_depth = 13,
     min_child_weight = 1,
     subsample = 0.9,
     colsample_bytree = 0.9)
```

#### [[10-2. 건물기타 모델링]]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/37f02275-a971-41d9-b62b-f06638a78c31)

- 사용변수: '태양광용량(kW)', '불쾌지수', '체감온도', '냉방효율', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.01,
     max_depth = 15,
     min_child_weight = 2,
     subsample = 0.9,
     colsample_bytree = 1.0)
```

#### [10-3. 대학교 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/afdfe00d-1781-40b7-869c-4fd74a4f72d4)

- 대학교의 경우 태양광용량이 존재하지 않아 해당 변수칸이 비어있는 것을 확인할 수 있음
- 사용변수: '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning은 아래 코드 참고
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 1500,
     learning_rate = 0.01,
     max_depth = 13,
     min_child_weight = 3,
     subsample = 0.9,
     colsample_bytree = 1.0)
```

#### [10-4. 데이터센터 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/fa5c4826-e9df-4c99-9ca0-1db9fbc93551)

- 사용변수: '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: RandomForestRegressor
- Parameter Tuning은 진행하지 않음
```
model = RandomForestRegressor()
```

#### [10-5. 백화점및아울렛 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/e8c44f66-6b7b-4579-ba69-cabb0b299f8b)

- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 1500,
     learning_rate = 0.01,
     max_depth = 13,
     min_child_weight = 3,
     subsample = 0.9,
     colsample_bytree = 1.0)
```

#### [10-6. 병원 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/b4f45f86-67e9-4b04-a2ef-330ed4b0f2e2)

- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.05,
     max_depth = 19,
     min_child_weight = 3,
     subsample = 0.8,
     colsample_bytree = 1.0)
```

#### [10-7. 상용 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/072bf0b0-853c-4260-ab05-94768a040c2c)

- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 1500,
     learning_rate = 0.03,
     max_depth = 17,
     min_child_weight = 2,
     subsample = 1.0,
     colsample_bytree = 1.0)
```

#### [10-8. 아파트 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/5354d4c9-987d-4909-9659-7e4b7f0e2bd3)

- 사용변수: '냉방효율', 'sin_time', 'cos_time', '휴일', '월', '일', '기온(C)', '풍속(m/s)', '습도(%)'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
- ```
  # 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.05,
     max_depth = 19,
     min_child_weight = 3,
     subsample = 0.8,
     colsample_bytree = 1.0)
  ```

#### [10-9. 연구소 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/2c564482-55c6-44aa-881e-2b8a002cdef5)

- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.01,
     max_depth = 15,
     min_child_weight = 2,
     subsample = 0.9,
     colsample_bytree = 1.0)
```

#### [10-10. 지식산업센터 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/740d5362-004b-43ee-87b8-26139cc26d45)

- 사용변수: '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.05,
     max_depth = 19,
     min_child_weight = 3,
     subsample = 0.8,
     colsample_bytree = 1.0)
```

#### [10-11. 할인마트 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/93edb011-df5c-417f-bec3-d2bba11dfa3e)
- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 1500,
     learning_rate = 0.01,
     max_depth = 13,
     min_child_weight = 3,
     subsample = 0.9,
     colsample_bytree = 1.0)
```

#### [10-12. 호텔및리조트 모델링]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/ee78210d-844c-43dd-86f0-08e82eb971c3)

- 사용변수: '태양광용량(kW)', '불쾌지수', '냉방효율', '체감온도', 'sin_time', 'cos_time', '휴일', '월', '일'
- 사용모델: XGBOOSTRegressor
- Parameter Tuning
```
# 모델 종류 선정
model = xgb.XGBRegressor(
     n_estimators = 700,
     learning_rate = 0.05,
     max_depth = 19,
     min_child_weight = 3,
     subsample = 0.8,
     colsample_bytree = 1.0)
```
