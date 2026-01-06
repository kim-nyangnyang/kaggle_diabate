# kaggle_diabate

# 📖 프로젝트 주제 : 당뇨병 예측 모델링: 통계분석 및 머신러닝 접근
- 머신러닝 기반 당뇨병 진단 예측 모델
  
 **"당뇨병은 완치가 어려운 만성질환이기에 '관리와 예방'이 무엇보다 중요합니다."**

당뇨병은 초기 증상이 뚜렷하지 않아 진단 시기를 놓치는 경우가 많습니다. 본 프로젝트는 데이터 분석을 통해 **고위험군을 정밀하게 식별**하고, 의료 자원의 효율적 배분을 돕는 **예방 의학적 솔루션**을 제안하는 데 목적이 있습니다. 분석을 통해 고위험군을 선별해낼 수 있다면, 환자 스스로 조심하게 만드는 예방적 가치가 매우 큽니다.

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

### 💡 주요 분석 포인트
1. **Target Variable**: 본 데이터셋의 목적은 다양한 변수를 통해 당뇨 발병 가능성을 예측하는 것입니다.
2. **Feature Importance**: 당뇨 진단 여부에 영향을 주는 중요도를 분석해 핵심요인을 도출합니다.
3. **Pre-processing**: 범주형 데이터(`gender`, `smoking_status` 등)는 모델 학습을 위해 One-Hot Encoding 또는 Label Encoding 처리가 필요합니다. 수치형 데이터(`bmi`,`systolic_bp` 등)는 StandardScaler, log변환 등 표준화가 필요합니다.

---


### 🔍 머신러닝에 사용한 변수

| 분류 | 변수명 | 설명 | 비고 (통계/처리 근거) |
| :--- | :--- | :--- | :--- |
| **인구통계** | `age` | 대상자의 연령 | 효과크기 0.33 (주요 인자) |
| **생활습관** | `log_physical_activity` | 주당 신체 활동 시간 | Log 변환, 효과 크기 0.34 |
| | `diet_score` | 평소 식습관 자가 점수 | 효과 크기 0.1  |
| **신체지표** | `bmi` | 체질량 지수 | 비만도 측정 (WHR 대체) |
| | `systolic_bp` | 수축기 혈압 | 혈압 지표 대표 |
| | `log_triglycerides` | 중성지방 수치 | Log 변환, 인슐린 저항성 지표 |
| | `hdl_cholesterol` | HDL 콜레스테롤 | 지질 대사 보호 요인 |
| **기저질환** | `family_history_diabetes` | 당뇨 가족력 여부 | Cohen's $d$ 0.44 (최고치) |
| **사회/환경** | `income_level` | 소득 수준 | 사회경제적 일반화 변수 |
| | `education_level` | 최종 학력 수준 | 건강 문해력 대리 지표 |
| **파생변수** | `age_family_interaction` | 연령 x 가족력 상호작용 | 노화에 따른 유전적 소인 발현 가중치 반영 |
| | `tg_hdl_ratio` | TG / HDL 비율 | 인슐린 저항성을 나타내는 핵심 임상 지표 |


### 💡 선택 기준 (Feature Selection Rationale)

1. **다중공선성(Multicollinearity) 해결**
   - VIF(분산팽창지수)가 극도로 높았던 `waist_to_hip_ratio`, `cholesterol_total`, `diastolic_bp` 등을 제외하고, 각 카테고리를 대표하는 핵심 변수만을 선정하여 모델의 통계적 안정성을 확보했습니다.

2. **통계적 유의성 및 효과 크기(Effect Size) 반영**
   - 데이터가 크기때문에 p-value뿐만 아니라 실질적 영향력을 나타내는 **Cohen's $d$** 지표를 기준으로 변수를 선별하였습니다.

3. **모델의 일반화(Generalization) 성능 향상**
   - 생물학적 수치 외에도 소득 및 교육 수준과 같은 사회경제적 지표를 포함하여, 개인의 환경적 맥락이 당뇨 발병에 미치는 비선형적 상호작용을 모델이 학습할 수 있도록 설계했습니다.

4. **데이터 분포 최적화**
   - 왜도(Skewness)가 높은 `triglycerides`와 `physical_activity`에 **Log 변환**을 적용하여 수치 범위 차이로 인한 왜곡을 방지하고 학습 효율을 높였습니다.
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



## 4. Data Preprocessing
- **클래스 불균형 해소** 
    + 타겟 변수(`diagnosed_diabetes`)의 비대칭적 분포 확인
    + 학습 시 클래스 가중치(Class Weight)를 조정하여, 소수 클래스인 당뇨 판정 오류에 대해 더 높은 페널티를 부여함으로써 예측 정확도와 재현율 간의 균형을 도모
- **범주형 변수 처리**
    + 순서형 : ordinal encoder 처리 (A, B, C)
    + 일반 범주 : One-Hot Encoding 처리
- **데이터 스케일링** : StandardScaler(표준화)
- **Log변환** : 불균형한 분포 대상 log 변환


## 5. 통계분석 핵심 인사이트

본 프로젝트는 모델 학습 전, 데이터의 통계적 구조를 파악하여 모델의 신뢰성을 확보했습니다.

### ✅ 다중공선성(VIF) 정제 결과
- **문제 진단**: 초기 분석 시 `waist_to_hip_ratio`(VIF: 817), `log_triglycerides`(VIF: 510) 등에서 극심한 다중공선성 확인
- **해결 전략**: 임상적 중요도가 겹치는 변수를 제거하고 대표 지표(`bmi`, `systolic_bp`)를 선정하여 VIF를 안정적인 수준으로 제어

### ✅ 효과 크기(Effect Size) 분석
단순 p-value 유의성을 넘어, 실제 당뇨 발병에 기여하는 정도를 **Cohen's $d$**와 **Cramér's $V$**로 정량화했습니다.

1. **가장 강력한 예측 요인**: **당뇨 가족력(0.44)**과 **신체 활동(-0.35)**이 압도적인 영향력을 보임
2. **신체 지표의 기여**: 연령(0.33) > 수축기 혈압(0.22) > BMI(0.21) 순으로 높은 상관관계 확인
3. **사회적 요인**: 소득 및 교육 수준이 미세하지만 유의미한 상관성을 보이며 모델의 일반화에 기여

### ✅ 파생 변수의 도입 근거
- **TG/HDL Ratio**: 중성지방과 HDL의 개별 수치보다 두 변수의 비율이 인슐린 저항성을 더 잘 설명한다는 임상 근거를 바탕으로 도입 고려
- **Age-Family Interaction**: 고연령층일수록 유전적 요인의 발현 가능성이 높아지는 비선형적 특성을 모델링에 반영



## 6. 모델링 평가지표
- 최종 모델은 LightGBM으로 선정

| Model | AUC-ROC | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression| 0.69 | 0.57 | 0.59 | 0.65 |
| **LightGBM** | **0.72** | **0.65** | **0.72** | **0.62** |

> **Note** : 최종 대회 결과는 Public 0.69515 / Private 0.69515 (상위 10%). 



## 7. Feature Importance
- SHAP 활용
- 예측 모델에서 영향력이 가장 컸던 지표 순위
1. **주당 신체 활동 시간 (`log_physical_activity`) - 0.40**: 모델의 판단에 가장 결정적인 기여를 하는 변수로, 활동량의 변화가 당뇨 예측에 가장 민감하게 작용함.
2. **연령 * 가족력 상호작용 (`age * family_history`) - 0.28**: 본 프로젝트에서 생성한 파생변수가 2위를 기록. 유전적 요인과 노화가 결합되었을 때 발병 위험이 급격히 상승하는 비선형적 특성을 모델이 정확히 포착함.
3. **연령 (`age`) - 0.23**: 생물학적 노화에 따른 기본적인 발병 위험 기여도 확인.
4. **중성지방 수치 (`log_triglycerides`) - 0.13**: 혈중 지질 농도가 인슐린 저항성을 설명하는 주요 지표로 활용됨.
5. **당뇨 가족력 여부 (`family_history_diabetes`) - 0.11**: 유전적 소인 단독으로도 유의미한 예측 성능을 보임.
![sharp]<<img width="789" height="339" alt="sharp" src="https://github.com/user-attachments/assets/b4523fbf-3a6d-4bcc-a278-7adee779746e" />


> **💡 인사이트**: 단순한 신체 지표(BMI 등)보다 **생활 습관(신체 활동)**과 **인구통계적 복합 요인(연령*가족력)**이 예측 모델에서 더 강력한 설명력을 가짐을 확인하였습니다.



## 8. Conclusion
### 🎯 프로젝트 성과 및 결론
- **예방 의학적 가치 실현**: 당뇨병은 완치보다 관리와 예방이 핵심인 질환입니다. 본 프로젝트는 방대한 진단 데이터를 바탕으로 고위험군을 사전에 선별할 수 있는 예측 지표를 구축하여, 의료 자원의 효율적 배분과 환자 스스로의 조기 관리를 돕는 예방 의학적 가치를 실현하고자 했습니다.
- **데이터 기반의 정교한 변수 선별**: VIF 분석을 통해 다중공선성을 유발하는 중복 변수(WHR, 총 콜레스테롤 등)를 제거하고, 핵심 지표 위주로 모델을 정예화하여 통계적 신뢰성을 확보하였습니다.
- **파생변수의 유효성 입증**: `age * family_history` 상호작용 변수가 기여도 2위를 차지함으로써, 단순 선형 관계가 아닌 복합적 위험 인자를 모델링한 것이 예측력 향상에 핵심적이었음을 증명하였습니다.
- **클래스 불균형 극복**: 약 9:1의 불균형 데이터셋 환경에서 `scale_pos_weight`와 `Class Weight` 조정을 적용하여, 정확도에 함몰되지 않고 실제 환자를 놓치지 않는 강건한 모델을 구축하였습니다.

### 💡 기대 효과 및 활용 방안

1. **개인 맞춤형 위험 알림**
   - SHAP 분석 결과 기여도가 높았던 **'신체 활동(Physical Activity)'**과 **'지질 수치(TG/HDL)'** 등을 실시간 모니터링하여, 사용자에게 개인화된 건강 관리 가이드를 제공할 수 있습니다.
   
2. **디지털 스크리닝 시스템 (Digital Screening)**
   - 병원 방문 전 단계에서 간단한 신체 지표 입력만으로 당뇨 위험도를 예측합니다. 이는 증상이 나타나기 전 **조기 발견 및 치료**를 유도하는 1차 스크리닝 도구로 활용 가능합니다.
   
3. **의료 자원 효율화**
   - 예측 모델을 통해 선별된 고위험군에게 의료 서비스를 집중함으로써, 사회적 비용을 절감하고 공공보건의 효율성을 높일 수 있습니다.
   - 
### 🚀 향후 과제 및 개선 방향
- **추가 지표 도입**: 현재 데이터셋 외에 식단의 세부 영양소 구성이나 스트레스 지수 등 정성적 생활 습관 데이터를 보완한다면 모델의 정교함을 더욱 높일 수 있을 것으로 기대됩니다.
- **모델 경량화**: 현재 가장 우수한 성능을 보인 LightGBM/XGBoost 모델을 기반으로, 모바일 헬스케어 기기 등 실제 임상 현장에서 실시간으로 구동 가능한 수준의 경량화 연구가 필요합니다.



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
