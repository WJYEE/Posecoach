# PoseCoach - AI 스쿼트 자세 분석기

AI 기반 실시간 스쿼트 자세 교정 및 카운팅 시스템입니다. MediaPipe와 딥러닝을 활용하여 정확한 자세 분석과 피드백을 제공합니다.

## 주요 기능

- 🎯 **실시간 자세 분석**: MediaPipe를 활용한 33개 관절점 추적
- 🔢 **자동 카운팅**: 정확한 스쿼트 동작 감지 및 카운팅
- 📊 **5단계 자세 분류**: 시작자세, 정상자세, 무릎문제, 골반문제, 복합문제
- 🔊 **음성 피드백**: 실시간 한국어 음성 가이드
- 📈 **성과 추적**: CSV 파일로 운동 기록 저장
- 🎨 **직관적 UI**: 실시간 관절점 시각화 및 피드백

## 시스템 요구사항

- Python 3.8 이상
- 웹캠
- Windows/macOS/Linux
- 최소 4GB RAM

## 설치 및 실행 가이드

### 1. 저장소 다운로드

```bash
git clone <repository-url>
cd PoseCoach
```

### 2. Anaconda 가상환경 생성

```bash
# Anaconda Prompt 실행 후
conda create -n posecoach python=3.9
conda activate posecoach
```

### 3. 필요한 패키지 설치

```bash
pip install opencv-python mediapipe tensorflow gtts pillow playsound joblib scikit-learn
```

### 4. 실행

```bash
cd src
python main.py
```

## 사용법

### 기본 조작

- **시작**: `python main.py` 실행
- **종료**: `q` 키 누르기
- **리셋**: `r` 키로 세션 초기화

### 운동 방법

1. 카메라 앞에 서서 전신이 보이도록 위치 조정
2. "시작 자세를 유지하세요" 메시지 확인
3. 정상적인 스쿼트 동작 수행
4. 실시간 피드백에 따라 자세 교정
5. 10회 완료 시 자동으로 다음 세트 진행

### 자세 분류

- **시작자세(0)**: 서있는 기본 자세
- **정상자세(1)**: 올바른 스쿼트 자세
- **무릎문제(2)**: 무릎이 안쪽으로 모이는 문제
- **골반문제(3)**: 허리가 굽거나 골반이 기울어진 문제
- **복합문제(4)**: 여러 문제가 동시에 발생

## 프로젝트 구조

```
PoseCoach/
├── src/
│   ├── main.py                              # 메인 실행 파일
│   ├── best_mlp_xgboost_replacement.keras   # 훈련된 AI 모델
│   ├── scaler.pkl                           # 특징 스케일러
│   ├── beep.wav                             # 경고음 파일
│   ├── MALGUN.TTF                           # 한글 폰트
│   └── tts_cache/                           # TTS 음성 캐시 폴더
├── data/                                    # 훈련 데이터 (선택사항)
│   ├── weighted_features_train_val_160.npy
│   ├── weighted_labels_train_val_160.npy
│   ├── weighted_features_test_160.npy
│   └── weighted_labels_test_160.npy
├── squat_scores.csv                         # 운동 결과 기록
└── README.md
```

## 파일 설명

### 핵심 파일

- **`main.py`**: 메인 실행 스크립트, 실시간 포즈 분석 및 UI
- **`best_mlp_xgboost_replacement.keras`**: 160차원 특징 기반 훈련된 딥러닝 모델
- **`scaler.pkl`**: 특징 정규화를 위한 StandardScaler

### 지원 파일

- **`beep.wav`**: 잘못된 자세 시 재생되는 경고음
- **`MALGUN.TTF`**: 한글 텍스트 표시용 폰트 (Windows)
- **`squat_scores.csv`**: 운동 세션별 결과 자동 저장

### 데이터 파일 (훈련용)

- **`weighted_features_*_160.npy`**: 160차원 특징 벡터
- **`weighted_labels_*_160.npy`**: 5클래스 라벨 데이터

## 기술 스택

### AI/ML

- **MediaPipe**: 실시간 포즈 추출 (33개 관절점)
- **TensorFlow**: 딥러닝 모델 추론
- **scikit-learn**: 데이터 전처리 및 스케일링

### 컴퓨터 비전

- **OpenCV**: 웹캠 영상 처리 및 시각화
- **NumPy**: 수치 연산 및 특징 추출

### 사용자 인터페이스

- **PIL (Pillow)**: 한글 텍스트 렌더링
- **gTTS**: 실시간 음성 피드백 생성
- **playsound**: 오디오 재생

## 특징 추출 (160차원)

### 기본 특징 (132차원)

- 33개 관절점 × (x, y, z, visibility)

### 추가 특징 (28차원)

- **측면 각도**: 목, 머리, 팔, 상체, 다리 각도 (11차원)
- **품질 점수**: 자세 안정성 평가 (4차원)
- **대칭성**: 좌우 균형 분석 (5차원)
- **거리**: 관절 간 거리 측정 (4차원)
- **각도**: 추가 관절 각도 (4차원)

## 성능 최적화

- **프레임 스킵**: 3프레임마다 분석으로 실시간성 확보
- **예측 안정화**: 다수결 방식으로 노이즈 제거
- **메모리 효율**: 최소한의 버퍼로 메모리 사용량 최적화
- **빠른 특징 추출**: 필수 계산만으로 지연시간 최소화

## 문제 해결

### 웹캠 문제

```python
# 웹캠 장치 번호 변경
cap = cv2.VideoCapture(1)  # 0 대신 1, 2 시도
```

### 폰트 문제

- Windows: `MALGUN.TTF` 파일 확인
- macOS/Linux: 시스템 폰트 자동 감지

### 성능 문제

- CPU 사용률 높을 시 프레임 스킵 주기 조정
- 조명 환경 개선으로 포즈 인식 정확도 향상

## 라이선스

본 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여하기

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 연락처

프로젝트 관련 문의나 버그 리포트는 이슈를 통해 등록해 주세요.

---

**PoseCoach** - AI가 도와주는 완벽한 스쿼트 자세 🏋️‍♀️
