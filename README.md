# kaggle_diabate


# 📖 프로젝트 주제 : 당뇨병 예측 모델링: 통계분석 및 머신러닝 접근
- 머신러닝 기반 당뇨병 진단 예측 모델
- 본 분석은 임상 데이터를 활용하여 당뇨병 발병의 핵심 위험 요인을 통계적으로 규명하고, 비선형적 상호작용을 효과적으로 포착하는 머신러닝 모델을 구축하는 것을 목적으로 합니다.

## 1. Project Overview 
- **주제** : 생활 습관 또는 신체 상태를 활용한 당뇨병 유무 분류
- **데이터셋** : [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)
- **핵심 목표** : 데이터를 활용해 **당뇨병 고위험군을 선별할 수 있는 예측 모델** 구축



## 2. Data Dictionary (주요 핵심 변수)
- 실제 분석 결과를 통해서 확보한 변수들의 기재
- 총 변수갯수 : 31개

## 📊 데이터셋 명세 (Dataset Specifications)

본 프로젝트에서 사용된 데이터는 Kaggle의 당뇨병 예측 챌린지 데이터를 기반으로 하며, 변수의 특성에 따라 5가지 카테고리로 분류하였습니다.

### 🔍 변수 정의 및 상세 설명

| 분류 | 변수명 | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **인구통계** | `age` | 대상자의 연령 | `Numeric` | 핵심 분석 지표 |
| | `gender` | 성별 (Male / Female) | `Categorical` | |
| | `ethnicity` | 인종 및 민족 | `Categorical` | |
| **생활습관** | `smoking_status` | 흡연 상태 | `Categorical` | Never, Former, Current |
| | `alcohol_consumption_per_week` | 주간 음주량 | `Numeric` | units/week |
| | `physical_activity` | 주당 신체 활동 시간 | `Numeric` | minutes/week |
| | `sleep_hours` | 일평균 수면 시간 | `Numeric` | hours/day |
| | `diet_score` | 평소 식습관 자가 점수 | `Numeric` | 1(불량) ~ 10(우수) |
| **신체지표** | `bmi` | 체질량 지수 (Body Mass Index) | `Numeric` | $kg/m^2$ |
| | `waist_to_hip_ratio` | 복부 비만도 (WHR) | `Numeric` | 허리/엉덩이 비율 |
| | `systolic_bp` | 수축기 혈압 | `Numeric` | mmHg |
| | `diastolic_bp` | 이완기 혈압 | `Numeric` | mmHg |
| | `cholesterol_total` | 총 콜레스테롤 수치 | `Numeric` | mg/dL |
| | `heart_rate` | 심박수 | `Numeric` | bpm |
| |`triglycerides` | 중성지방 | `Numeric` | mg/dL |
| |`hdl_cholesterol` | HDL 콜레스테롤(고밀도) | `Numeric` | mg/dL |
| |`ldl_cholesterol` | LDL 콜레스테롤(저밀도) | `Numeric` | mg/dL |
| **기저질환** | `family_history_diabetes`| 당뇨 가족력 여부 | `Binary` | 0: 없음, 1: 있음 |
| | `hypertension_history` | 고혈압 과거력 | `Binary` | 0: 없음, 1: 있음 |
| | `cardiovascular_history`| 심혈관 질환 과거력 | `Binary` | 0: 없음, 1: 있음 |
| **사회/환경** | `income_level` | 소득 수준 (Low ~ High) | `Ordinal` | |
| | `education_level` | 최종 학력 수준 | `Ordinal` | |
| | `employment_status` | 고용 형태 | `Ordinal` | |
| | `screen_time_hours_per_day` | 일일 스크린타임 | `Numeric` | hours/day | |
| **진단지표** | `diabetes_stage` | 당뇨병 진행 단계 | `Categorical` | 질환의 심각도 단계 | |
| | `diabetes_risk_score` | 당뇨병 위험 점수 | `Numeric` | 예측 모델의 기반 점수 | |
| | `hba1c` | **당화혈색소** | `Numeric` | % | |
| | `glucose_fasting` | 공복 혈당 | `Numeric` | mg/dL | |
| | `glucose_postprandial` | 식후 혈당 (2시간) | `Numeric` | mg/dL | |
| | `insulin_level` | 인슐린 수치 | `Numeric` | $\mu U/mL$ | |
| **targetVariable** | **`diagnosed_diabetes`** | **당뇨 진단 여부 (Target)** | `Binary` | **0: 음성, 1: 확진** |

---

### 💡 주요 분석 포인트
1. **Target Variable**: 본 데이터셋의 목적은 다양한 변수를 통해 당뇨 발병 가능성을 예측하는 것입니다.
2. **Feature Importance**: 
3. **Pre-processing**: 범주형 데이터(`gender`, `smoking_status` 등)는 모델 학습을 위해 One-Hot Encoding 또는 Label Encoding 처리가 필요합니다. 수치형 데이터(`bmi`,`systolic_bp` 등)는 StandardScaler등 표준화가 필요합니다.



## 3. Problem Definition
- **데이터 특성** 
    1. 복합적 변수 구성 : 응답자의 특성을 다양한 독립변수로 나타냄
    2. 수치형과 범주형의 혼재 : 전처리 필수
    3. 비선형적 관계 가능성 : 연령, 혈압, bmi 등 복합적 상호작용의가능성
    4. 다중공산성 : 변수 간 상관관계가 높아 다중 공선성 문제 존재 가능 
- **분석 방향**
    + 통계분석 : 다중회귀, 분산분석, 로지스틱회귀, 단변량 분석 등
    + 머신러닝 : 로지스틱회귀, 결정트리, XGBoost, LightGBM  등



## 4. Data Preprocessing
- **클래스 불균형 해소** 
    + 타겟 변수(`diagnosed_diabetes`)의 비대칭적 분포(약 9:1) 확인
    + 학습 시 클래스 가중치(Class Weight)를 조정하여, 소수 클래스인 당뇨 판정 오류에 대해 더 높은 페널티를 부여함으로써 예측 정확도와 재현율 간의 균형을 도모
- **범주형 변수 처리**
    + 순서형 : ordinal encoder 처리 (A, B, C)
    + 일반 범주 : One-Hot Encoding 처리
- **데이터 스케일링** : StandardScaler(표준화)



## 5. 통계분석 핵심 인사이트
- 혈당이 중요함 : 다른 알려진 요인(나이, BMI)보다 통계적으로 매우 훨씬, 강력하게, 유의미하게 영향이 있음을 확인 (via 회귀분석)
![Q-Q Plot](output/qqplot.jpg)



## 6. 모델링 평가지표
- 최종 모델은 LightGBM으로 선정

| Model | AUC-ROC | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression| 0.69 | 0.62 | 0.59 | 0.66 |
| **LightGBM** | **0.72** | **0.65** | **0.63** | **0.69** |

> **Note** : 최종 대회 결과는 Public 0.70807 / Private 0.70807 (feat. 1등 점수). 

> **Note** : 최종 대회 결과는 Public 0.69515 / Private 0.69515 (상위 10%). 



## 7. Feature Importance (옵션)
- SHAP 활용
- 예측 모델에서 영향력이 가장 컸던 지표 순위
1. AGE
2. BMI
- 그림 추가



## 8. Conclusion
- 결론1
- 결론2
- 결론3



# 보고서
- 프로젝트 상세 보고서는 PDF 슬라이드 자료를 참고하여 주세요
- 00 보고서 : [당뇨병 예측 모델링: 통계분석 및 머신러닝 접근](report/프로젝트보고서.pdf)
- 분석코드 : [분석코드](report/프로젝트251230.ipynb)

# 🔗 배지 및 이모지 공식 소스 링크
| 용도 | 사이트 이름 | 링크 |
| :--- | :--- | :--- |
| **배지 생성** | Shields.io | [https://shields.io/](https://shields.io/) |
| **로고/색상 검색** | Simple Icons | [https://simpleicons.org/](https://simpleicons.org/) |
| **이모지 검색** | Emoji Cheat Sheet | [https://github.com/ikatyang/emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet) |