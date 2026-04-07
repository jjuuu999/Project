# Financial News Sentiment Analysis (NLP)

## Overview
금융 뉴스 텍스트의 감성을 **Positive / Neutral / Negative** 3개 클래스로 분류하는 딥러닝 모델 비교 프로젝트

## Dataset
- **FinancialPhraseBank** (Malo et al., 2014)
- Kaggle: [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- 4,846개 금융 뉴스 문장
- 클래스 분포: Neutral(2,879) / Positive(1,363) / Negative(604)
- Train: 3,876 / Test: 970 (80:20, Stratified)

## Models & Results

| Model | Test Accuracy | Test Loss | 비고 |
|-------|:---:|:---:|------|
| MLP | 68.35% | 0.7393 | 베이스라인, 단어 순서 무시 |
| **CNN** | **73.09%** | **0.7194** | **최고 성능 - 채택** |
| LSTM | 59.38% | 0.9262 | 학습 실패 (클래스 불균형) |
| GRU | 59.38% | 0.9261 | 학습 실패 (클래스 불균형) |

## Pipeline

```
1. 데이터 로드 및 EDA
2. Train/Test 분리 (stratify 적용)
3. 토크나이징 & 패딩 (padding='post', MAX_LEN=100)
4. 라벨 원-핫 인코딩
5. 4개 모델 학습 (EarlyStopping, patience=5, 모델별 개별 객체 생성)
6. 모델 비교 → CNN 채택
7. 실제 Yahoo Finance 뉴스 기사로 예측 (본문 전체 / 앞 3문장)
```

## Key Findings

- **CNN이 금융 텍스트 감성분석에 가장 적합** — 핵심 키워드 패턴("profit increased", "loss reported") 추출에 강점
- **LSTM/GRU 학습 실패** — 데이터 규모(4,846건)가 작고 클래스 불균형(neutral 59%)으로 다수 클래스만 예측하는 현상 발생
- **EarlyStopping 주의사항** — 모델마다 새로운 객체를 생성해야 내부 카운터가 초기화됨 (patience=5 적용)
- **패딩 전략** — `post` 패딩 사용. 금융 뉴스 특성상 문장 뒷부분(but, however 이후)이 감성을 뒤집는 경우가 많아 LSTM/GRU에서는 `pre` 패딩이 더 유리할 수 있음

## 개선 가능한 점

- GloVe 사전학습 임베딩 적용 (v2에서 구현)
- Bidirectional LSTM/GRU로 양방향 문맥 파악
- class_weight 적용으로 클래스 불균형 해결
- SHAP/LIME을 활용한 예측 근거 시각화 (XAI)
- F1-score, Confusion Matrix 등 추가 평가지표

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- scikit-learn
- NLTK
- newspaper3k
