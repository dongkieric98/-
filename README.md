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
  
## 6. 건물정보 데이터 전처리
#### [건물정보 데이터 결측치 비율]
![image](https://github.com/dongkieric98/Electricity_Consumption_Forecasting_Project/assets/118495885/aac0b9d9-9d06-4551-afa3-a58176ce3b80)


#### [태양광용량 전처리리
