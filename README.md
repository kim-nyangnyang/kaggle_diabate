# kaggle_diabate

# 📖 당뇨병 예측 모델링: 통계분석 및 머신러닝 접근
- 머신러닝 기반 당뇨병 진단 예측 모델
  
 **"당뇨병은 완치가 어려운 만성질환이기에 '관리와 예방'이 무엇보다 중요합니다."**

당뇨병은 초기 증상이 뚜렷하지 않아 진단 시기를 놓치는 경우가 많습니다. 본 프로젝트는 데이터 분석을 통해 **고위험군을 정밀하게 식별**하고, 의료 자원의 효율적 배분을 돕는 **예방 의학적 솔루션**을 제안하는 데 목적이 있습니다. 분석을 통해 고위험군을 선별해낼 수 있다면, 환자 스스로 조심하게 만드는 예방적 가치가 매우 큽니다.

## 1. Project Overview 
- **주제** : 생활 습관 또는 신체 상태를 활용한 당뇨병 유무 분류
- **데이터셋** : [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)
- **핵심 목표** : 
  1. 다중공선성(VIF) 문제를 해결한 신뢰성 있는 변수 선정
  2. 클래스 불균형(Class Imbalance)을 극복한 고위험군 탐지 모델 구축
  3. SHAP 분석을 통한 주요 발병 인자 도출 및 해석



## 2. Data Dictionary (주요 핵심 변수)
- 실제 분석 결과를 통해서 확보한 변수들의 기재
- 총 변수갯수 : 31개

## 📊 데이터셋 명세 (Dataset Specifications)

본 프로젝트에서 사용된 데이터는 Kaggle의 당뇨병 예측 챌린지 데이터를 기반으로 하며, 변수의 특성에 따라 5가지 카테고리로 분류하였습니다.

### 🔍 변수 정의 및 상세 설명

| 분류 | 변수명 | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **인구통계** | `age` | 대상자의 연령 | `Numeric` | 핵심 분석 지표 |
| | `gender` | 성별 (Male / Female) | `Categorical` | 남성,여성, other |
| | `ethnicity` | 인종 및 민족 | `Categorical` |white 등 5개 |
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
| **사회/환경** | `income_level` | 소득 수준  | `Ordinal` |(Low ~ High) |
| | `education_level` | 최종 학력 수준 | `Ordinal` | Graduate 등 4개 |
| | `employment_status` | 고용 형태 | `Ordinal` | employment 등 4개 |
| | `screen_time_hours_per_day` | 일일 스크린타임 | `Numeric` | hours/day | |
| **진단지표** | `diabetes_stage` | 당뇨병 진행 단계 | `Categorical` | 질환의 심각도 단계 | |
| | `diabetes_risk_score` | 당뇨병 위험 점수 | `Numeric` | 예측 모델의 기반 점수 | |
| | `hba1c` | **당화혈색소** | `Numeric` | % | |
| | `glucose_fasting` | 공복 혈당 | `Numeric` | mg/dL | |
| | `glucose_postprandial` | 식후 혈당 (2시간) | `Numeric` | mg/dL | |
| | `insulin_level` | 인슐린 수치 | `Numeric` | $\mu U/mL$ | |
| **targetVariable** | **`diagnosed_diabetes`** | **당뇨 진단 여부 (Target)** | `Binary` | **0: 음성, 1: 확진** |
---
## 3. Problem Definition
- **데이터 특성** 
    1. 복합적 변수 구성 : 응답자의 특성을 다양한 독립변수로 나타냄
    2. 수치형과 범주형의 혼재 : 전처리 필수
    3. 비선형적 관계 가능성 : 연령, 혈압, bmi 등 복합적 상호작용의가능성
    4. 다중공산성 : 변수 간 상관관계가 높아 다중 공선성 문제 존재 가능 
- **분석 방향**
    + 통계분석 : 다중회귀, 분산분석, 로지스틱회귀, 단변량 분석 등
    + 머신러닝 : 로지스틱회귀, 결정트리, XGBoost, LightGBM  등

## 4. 🛠️ Data Preprocessing
- **클래스 불균형 해소**: 타겟 변수(`diagnosed_diabetes`)의 비대칭적 분포를 해결하기 위해 `scale_pos_weight` 및 `Class Weight` 조정 적용.
- **데이터 스케일링**: 수치형 데이터에 `StandardScaler` 적용 및 왜도가 높은 변수(`physical_activity`)에 `Log 변환` 수행.
- **범주형 변수 처리**: 순서형 변수는 `Ordinal Encoding`, 일반 범주형은 `One-Hot Encoding` 적용.

## 5. 통계분석 핵심 인사이트
본 프로젝트는 모델 학습 전, 데이터의 통계적 구조를 파악하여 모델의 신뢰성을 확보했습니다.

### ✅ 다중공선성(VIF) 정제 결과
- **문제 진단**: 초기 분석 시 `waist_to_hip_ratio`(VIF: 817), `log_triglycerides`(VIF: 510) 등에서 극심한 다중공선성 확인
- **해결 전략**: 임상적 중요도가 겹치는 변수를 제거하고 대표 지표(`bmi`, `systolic_bp`)를 선정하여 VIF를 안정적인 수준으로 제어

### ✅ 효과 크기(Effect Size) 분석
단순 p-value 유의성을 넘어, 실제 당뇨 발병에 기여하는 정도를 **Cohen's $d$**와 **Cramér's $V$**로 정량화했습니다.

### ✅ 파생 변수의 도입
- **TG/HDL Ratio**: 중성지방과 HDL의 개별 수치보다 두 변수의 비율이 인슐린 저항성을 더 잘 설명한다는 임상 근거를 바탕으로 도입 고려
- **Age-Family Interaction**: 고연령층일수록 유전적 요인의 발현 가능성이 높아지는 비선형적 특성을 모델링에 반영
---

### 🔍 머신러닝에 사용한 변수
초기 31개의 변수 중 분석 결과에 따라 모델의 성능과 해석력을 높이는 핵심 변수 위주로 정제하였습니다.


| 분류 | 변수명 | 설명 | 비고 (통계/처리 근거) |
| :--- | :--- | :--- | :--- |
| **기저질환** | `family_history_diabetes` | 당뇨 가족력 여부 | **Cohen's d: 0.44 (최고치)**, 강력한 예측 인자 |
| **생활습관** | `log_physical_activity` | 주당 신체 활동 시간 | Log 변환 적용, **Cohen's d: 0.34** |
| **인구통계** | `age` | 대상자의 연령 | **Cohen's d: 0.33**, 노화에 따른 위험 통제 |
| **대사지표** | `TG/HDL` | 중성지방/HDL 비율 | **파생변수**, 인슐린 저항성 핵심 지표 (VIF 해결) |
| **신체지표** | `bmi` | 체질량 지수 (BMI) | 비만도 측정 대표 지표 (WHR 대체) |
| **혈관건강** | `systolic_bp` | 수축기 혈압 | 혈압 상태 반영 (다중공선성 고려 선정) |
| **사회경제** | `income_level` | 소득 수준 | 사회경제적 일반화 및 맥락 반영 |
| **기저질환** | `cardiovascular_history` | 심혈관 질환 과거력 | 고위험군 식별을 위한 레드 플래그 |


### 💡 선택 기준 (Feature Selection Rationale)

1. **다중공선성(Multicollinearity) 해결**: VIF가 극도로 높았던 `waist_to_hip_ratio`(817), `log_triglycerides`(517) 등을 제거하고, 이를 함축하는 `bmi` 및 `TG/HDL` 비율을 사용하여 통계적 안정성을 확보했습니다.
2. **효과 크기(Effect Size) 반영**: 단순 p-value를 넘어 실제 영향력을 나타내는 **Cohen's d** 지표를 기준으로 변수를 선별하였습니다.
3. **모델의 일반화(Generalization) 성능 향상**: 사회경제적 지표를 포함하여, 개인의 환경적 맥락이 당뇨 발병에 미치는 비선형적 상호작용을 모델이 학습할 수 있도록 설계했습니다.
---

## 6. 모델링 평가지표
최종 모델은 불균형 데이터 처리와 비선형 관계 학습에 탁월한 **LightGBM**을 선정하였습니다.

| Model | AUC-ROC | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression| 0.69 | 0.57 | 0.59 | 0.65 |
| **LightGBM** | **0.72** | **0.65** | **0.72** | **0.62** |

> **Note** : 최종 대회 결과는 Public 0.69515 / Private 0.69515 (상위 10%). 



## 7. 🔍 Feature Importance (SHAP Analysis)
SHAP 분석을 통해 예측 모델이 각 개인을 판단할 때 중요하게 고려한 지표의 순위를 도출했습니다.
**[Top 5 핵심 인자]**
1. **당뇨 가족력 (`family_history_diabetes`) - 0.407**: 유전적 소인이 당뇨 발생을 예측하는 가장 압도적인 지표임이 확인되었습니다.
2. **신체 활동량 (`log_physical_activity`) - 0.401**: 활동량이 적을수록 발병 위험이 급격히 상승하며, 생활 습관 중 가장 결정적인 기여를 합니다.
3. **연령 (`age`) - 0.259**: 생물학적 노화에 따른 기본적인 발병 위험 기여도를 포착하였습니다.
4. **인슐린 저항성 지표 (`TG/HDL`) - 0.123**: 중성지방과 HDL의 비율이 단순 신체 지표보다 당뇨 예측에 더 민감하게 작용합니다.
5. **체질량 지수 (`bmi`) - 0.095**: 비만도가 물리적 위험 인자로서 유의미한 설명력을 가집니다.

> **💡 인사이트**: 단순한 신체 지표(BMI 등)보다 **가족력(유전)**과 **신체 활동(습관)**이 모델 예측에서 더 강력한 설명력을 가짐을 확인하였습니다. 특히 파생변수인 `TG/HDL` 비율이 주요 인자로 등장하며 대사 지표의 중요성을 입증했습니다.

---
![sharp]<img width="789" height="339" alt="sharp" src="https://github.com/user-attachments/assets/64daa793-bbe8-4d47-92b0-7d647f9dfc5f" />



## 8. Conclusion
### 🎯 프로젝트 성과 및 결론
- **예방 의학적 가치 실현**: 방대한 데이터를 바탕으로 고위험군을 사전에 선별할 수 있는 지표를 구축하여 조기 관리를 돕는 예방 의학적 모델을 구현했습니다.
- **데이터 기반의 정교한 변수 선별**: VIF 분석을 통해 다중공선성을 유발하는 중복 변수를 정제하고, 통계적으로 견고한 8개의 핵심 변수 모델을 완성했습니다.

### 💡 기대 효과 및 활용 방안

1. **개인 맞춤형 위험 알림**
   - SHAP 분석 결과 기여도가 높았던 **'신체 활동(Physical Activity)'**과 **'지질 수치(TG/HDL)'** 등을 실시간 모니터링하여, 사용자에게 개인화된 건강 관리 가이드를 제공할 수 있습니다.
   
2. **디지털 스크리닝 시스템 (Digital Screening)**
   - 병원 방문 전 단계에서 간단한 신체 지표 입력만으로 당뇨 위험도를 예측합니다. 이는 증상이 나타나기 전 **조기 발견 및 치료**를 유도하는 1차 스크리닝 도구로 활용 가능합니다.
   
3. **의료 자원 효율화**
   - 예측 모델을 통해 선별된 고위험군에게 의료 서비스를 집중함으로써, 사회적 비용을 절감하고 공공보건의 효율성을 높일 수 있습니다.



# 보고서
- 프로젝트 상세 보고서는 PDF 슬라이드 자료를 참고하여 주세요
- 분석 보고서 : [당뇨병 예측 모델링: 통계분석 및 머신러닝 접근](report/당뇨진단예측보고서.pdf)
- 분석코드 : [분석코드](report/기본틀.ipynb)
- Tableau: [대시보드 열기]([https://github.com/kim-nyangnyang/kaggle_diabate/blob/main/report/%EB%8B%B9%EB%87%A8.twb](https://public.tableau.com/app/profile/.75493083/viz/diabetes_tableau/14))

# 🔗 배지 및 이모지 공식 소스 링크
| 용도 | 사이트 이름 | 링크 |
| :--- | :--- | :--- |
| **배지 생성** | Shields.io | [https://shields.io/](https://shields.io/) |
| **로고/색상 검색** | Simple Icons | [https://simpleicons.org/](https://simpleicons.org/) |
| **이모지 검색** | Emoji Cheat Sheet | [https://github.com/ikatyang/emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet) |
