# 🎨 명화 vs AI 그림 분류기

사람이 그린 명화와 생성형 AI가 만든 그림을 구분하는 CNN 이미지 분류 모델과 Streamlit 웹 데모입니다.

> 동양미래대학교 인공지능소프트웨어학과 전공동아리 Mini-challenge 3조 프로젝트 (2025.04 ~ 2025.05, 4인 팀)

## 프로젝트 배경

생성형 AI 그림이 빠르게 늘어나면서 사람이 그린 작품과 구분하기 어려워졌습니다.
"AI 그림이 많아지는 시대, 진짜와 가짜를 구분할 수 있을까?"라는 질문에서 출발해,
이미지를 올리면 명화인지 AI 그림인지 판별하는 모델을 만들었습니다.

## 데이터

| 구분 | 내용 |
|---|---|
| 출처 | Kaggle (실제 명화 + Stable Diffusion·Latent Diffusion 생성 그림, 10개 미술 사조) |
| 확보 | 약 15만 장 |
| 학습 사용 | 명화 2,500장 + AI 그림 2,500장 = 5,000장 (무작위 샘플링) |

노트북 환경의 컴퓨팅 자원 한계 때문에, 클래스 균형을 맞춘 5,000장으로 먼저 빠르게 실험하는 방식을 택했습니다.
데이터셋은 용량 문제로 저장소에 포함하지 않았습니다.

## 처리 과정

```
데이터 수집 (Kaggle API)
   ↓
전처리: 파일명 기반 자동 라벨링 (명화 0 / AI 1), train·val·test 분할
   ↓
224×224 크기 통일, RGB 변환, 0~1 정규화
   ↓
CNN 학습 → 정확도·손실 그래프, 혼동 행렬, classification report로 평가
   ↓
Streamlit 웹 앱으로 실시간 분류
```

## 모델 구조

```
Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling
→ Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
```

- 손실 함수: binary crossentropy / 옵티마이저: Adam
- 출력: AI 그림일 확률 (0.5 기준으로 명화 / AI 그림 판정)

## 결과

- 검증 정확도 약 **86%**
- 혼동 행렬과 classification report에서 두 클래스의 정밀도·재현율이 균형을 이뤄, 한쪽 클래스로 치우치지 않음을 확인

## 폴더 구조

```
ai-art-classifier/
├── app.py                              # Streamlit 웹 앱
├── models/
│   └── train.py                        # CNN 모델 정의 및 학습
├── preprocessing/
│   ├── data_split.py                   # real/ai 폴더 → train/val/test 분할
│   └── data_split_from_filename.py     # 파일명 기반 라벨링 + 분할
└── requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 데이터 분할 (Real_AI_SD_LD_Dataset 폴더를 준비한 뒤)
python preprocessing/data_split_from_filename.py

# 2) 모델 학습 → simple_cnn_model.h5 생성
python models/train.py

# 3) 웹 앱 실행
streamlit run app.py
```

학습된 모델 파일(`.h5`)은 저장소에 포함하지 않았으므로, 먼저 2단계로 모델을 학습해야 합니다.

## 한계와 개선 방향

- 단순 CNN 구조라 추상적이거나 스타일이 섞인 그림에서는 헷갈릴 수 있습니다.
- 학습 데이터의 스타일 분포를 벗어난 새로운 유형의 AI 그림에는 약할 수 있습니다.
- 데이터 증강(회전, 노이즈, 색상 변형), 전이학습(EfficientNet, ViT) 적용, 학습 데이터 확대로 개선할 수 있습니다.

## 맡은 역할 (이종복)

- **데이터 수집**: Kaggle API로 명화·AI 그림 데이터셋 확보
- **전처리**: 파일명 기반 자동 라벨링, train·val·test 분할 스크립트 작성, 크기 통일·정규화
- **모델 개발**: CNN 분류 모델 설계·학습 및 성능 평가
- **웹 앱 개발**: Streamlit으로 이미지 업로드 → 실시간 분류 결과 화면 구현
- **발표**: 팀 대표 최종 발표 및 실시간 시연

## 기술 스택

Python, TensorFlow(Keras), NumPy, Pillow, Matplotlib, Streamlit, Jupyter Notebook, Kaggle API
