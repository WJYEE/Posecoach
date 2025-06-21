import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os, time, threading, csv
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
from playsound import playsound
import platform, joblib
import warnings
from collections import Counter

# 경고 및 로그 제거
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 경로 및 상수
TTS_DIR, BEEP_PATH, SCALER_PATH = "tts_cache", "beep.wav", "scaler.pkl"
MODEL_PATH = "best_mlp_xgboost_replacement.keras"
os.makedirs(TTS_DIR, exist_ok=True)
assert os.path.exists(MODEL_PATH), f"모델 파일 없음: {MODEL_PATH}"

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.7,  # 향상된 감지 신뢰도
    min_tracking_confidence=0.5,
    model_complexity=1  # 성능과 정확도 균형
)

def speak_async(text):
    """비동기 TTS 음성 재생 - 멀티스레딩으로 메인 루프 방해 없음"""
    def run():
        try:
            key = os.path.join(TTS_DIR, f"{hash(text)}.mp3")
            if not os.path.exists(key):
                gTTS(text=text, lang='ko').save(key)
            print(f"음성 재생: {text}")
            playsound(key)
        except Exception as e:
            print(f"음성 재생 오류: {e}")
    threading.Thread(target=run, daemon=True).start()

def play_beep():
    """잘못된 자세 시 경고음 재생"""
    try:
        threading.Thread(target=lambda: playsound(BEEP_PATH), daemon=True).start()
    except:
        print("비프음 파일 없음")

def simple_text(frame, text, pos, color=(0, 255, 255), size=20):
    """한글 텍스트 출력 - PIL 사용으로 인코딩 문제 해결"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 시스템별 한글 폰트 로드
        try:
            if platform.system() == "Windows":
                font = ImageFont.truetype("malgun.ttf", size)
            elif platform.system() == "Darwin":  # macOS
                font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", size)
            else:  # Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size)
        except:
            font = ImageFont.load_default()
            
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 폰트 실패 시 영문 대체
        text_en = text.replace("자세:", "Pose:").replace("피드백:", "FB:").replace("카운트:", "Count:")
        cv2.putText(frame, text_en, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def draw_landmarks_on_frame(frame, landmarks):
    """MediaPipe 관절점 시각화 - 더 선명한 색상"""
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    return frame

def safe_angle(a, b, c):
    """3점을 이용한 안전한 각도 계산 - 수치 안정성 보장"""
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        
        # 벡터 크기 계산
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 180.0  # 기본값
            
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)  # 수치 오류 방지
        return np.degrees(np.arccos(cosine))
    except:
        return 180.0

def vertical_angle(p1, p2):
    """수직선 기준 각도 계산"""
    try:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dy) < 1e-6:
            return 90.0
        return np.degrees(np.arctan(abs(dx) / abs(dy)))
    except:
        return 0.0

def extract_pose_features(image):
    """
    MediaPipe로 포즈 특징 추출 (성능 최적화 버전)
    - 33개 관절점에서 160개 특징 생성
    - 기본 좌표 + 핵심 각도 + 품질 점수 + 간소화된 추가 특징
    """
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark
    
    # 기본 특징: x, y, z, visibility (33 × 4 = 132개) - 최적화
    basic_features = []
    for pt in lm:
        basic_features.extend([pt.x, pt.y, pt.z, pt.visibility])
    
    # 핵심 관절점만 추출 (성능 최적화)
    try:
        # 간소화된 포인트 추출
        nose = [lm[0].x, lm[0].y]
        left_ear = [lm[7].x, lm[7].y]
        right_ear = [lm[8].x, lm[8].y]
        left_shoulder = [lm[11].x, lm[11].y]
        right_shoulder = [lm[12].x, lm[12].y]
        left_elbow = [lm[13].x, lm[13].y]
        right_elbow = [lm[14].x, lm[14].y]
        left_wrist = [lm[15].x, lm[15].y]
        right_wrist = [lm[16].x, lm[16].y]
        left_hip = [lm[23].x, lm[23].y]
        right_hip = [lm[24].x, lm[24].y]
        left_knee = [lm[25].x, lm[25].y]
        right_knee = [lm[26].x, lm[26].y]
        left_ankle = [lm[27].x, lm[27].y]
        right_ankle = [lm[28].x, lm[28].y]
        
        # 중앙점 계산 (간소화)
        center_ear = [(left_ear[0] + right_ear[0])/2, (left_ear[1] + right_ear[1])/2]
        center_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
        center_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
        
        # 핵심 측면 각도 특징 (11개) - 기존과 동일하지만 최적화
        lateral_features = [
            vertical_angle(center_ear, center_shoulder),                    # 목 기울기
            safe_angle(nose, center_ear, center_shoulder),                  # 머리 각도
            safe_angle(left_shoulder, left_elbow, left_wrist),              # 왼팔 각도
            left_shoulder[0] - center_ear[0],                               # 어깨-귀 수평 거리
            vertical_angle(center_shoulder, center_hip),                    # 상체 기울기
            safe_angle(center_shoulder, center_hip, left_knee),             # 엉덩이 각도
            safe_angle(center_hip, left_knee, left_ankle),                  # 왼쪽 무릎 각도
            vertical_angle(center_hip, left_knee),                          # 허벅지 기울기
            safe_angle(center_hip, right_knee, right_ankle),                # 오른쪽 무릎 각도
            vertical_angle(center_ear, left_ankle),                         # 전체 몸 기울기
            center_ear[0] - left_ankle[0]                                   # 머리-발목 수평 거리
        ]
        
        # 품질 점수 특징 (4개) - 기존과 동일
        quality_scores = [
            max(0, min(1, (lateral_features[1] - 120) / 60)),      # 머리 각도 품질
            max(0, min(1, (lateral_features[2] - 120) / 60)),      # 팔 각도 품질
            max(0, min(1, 1 - abs(lateral_features[4]) / 45)),     # 상체 기울기 품질
            max(0, min(1, 1 - abs(lateral_features[10]) / 0.35))   # 전체 균형 품질
        ]
        
        # 간소화된 추가 특징 (13개) - 계산 최적화
        additional_features = [
            # 대칭성 특징 (간소화, 5개)
            abs(left_shoulder[1] - right_shoulder[1]),                      # 어깨 높이 차이
            abs(left_hip[1] - right_hip[1]),                               # 골반 높이 차이
            abs(left_knee[1] - right_knee[1]),                             # 무릎 높이 차이
            abs(left_ankle[1] - right_ankle[1]),                           # 발목 높이 차이
            abs(center_shoulder[0] - center_hip[0]),                       # 몸통 중심 정렬
            
            # 핵심 거리 특징 (4개)
            abs(center_shoulder[1] - center_hip[1]),                       # 어깨-골반 세로 거리
            abs(center_hip[1] - left_knee[1]),                            # 골반-무릎 세로 거리
            abs(left_knee[1] - left_ankle[1]),                            # 무릎-발목 세로 거리
            abs(left_shoulder[0] - right_shoulder[0]),                     # 어깨 폭
            
            # 핵심 각도 특징 (4개)
            safe_angle(right_shoulder, right_elbow, right_wrist),          # 오른팔 각도
            safe_angle(center_shoulder, left_hip, left_knee),              # 왼쪽 몸통-다리 각도
            safe_angle(center_shoulder, right_hip, right_knee),            # 오른쪽 몸통-다리 각도
            vertical_angle(left_knee, left_ankle)                          # 발목 각도 (간소화)
        ]
        
    except Exception as e:
        print(f"특징 계산 오류: {e}")
        # 오류 시 기본값으로 빠르게 처리
        lateral_features = [0, 180, 180, 0, 0, 180, 180, 0, 180, 0, 0]
        quality_scores = [0.7, 0.7, 0.7, 0.7]
        additional_features = [0.0] * 13
    
    # 전체 특징 결합: 132 + 11 + 4 + 13 = 160개
    all_features = basic_features + lateral_features + quality_scores + additional_features
    
    # 160차원 보장 (빠른 처리)
    if len(all_features) != 160:
        if len(all_features) < 160:
            all_features.extend([0.0] * (160 - len(all_features)))
        else:
            all_features = all_features[:160]
    
    return np.array(all_features, dtype=np.float32).reshape(1, -1), results.pose_landmarks

class SquatSession:
    """
    스쿼트 운동 세션 관리 클래스
    - 카운팅, 점수 계산, 피드백 생성, 결과 저장
    """
    def __init__(self, user_id="user001"):
        # 기본 카운터
        self.count = 0                  # 현재 세트 카운트
        self.set_count = 0              # 완료된 세트 수
        self.perfect = 0                # 완벽한 자세 횟수
        self.max_per_set = 10           # 세트당 목표 횟수
        
        # 점수 및 라벨 추적
        self.score_log = []             # 실제 운동 점수만 기록
        self.exercise_labels = []       # 운동 시 라벨 기록
        self.label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 전체 라벨 빈도
        
        # 세션 정보
        self.user_id = user_id
        self.start_time = time.time()
        
        # 상태 관리
        self.prev_exercising = False    # 이전 프레임 운동 상태
        self.counting_cooldown = 0      # 중복 카운트 방지 쿨다운
        self.frame_count = 0            # 전체 프레임 수
        
        # 안정성 향상을 위한 버퍼
        self.recent_predictions = []    # 최근 예측 결과
        self.pose_buffer = []           # 포즈 안정화 버퍼

    def get_time(self):
        """경과 시간 포맷팅 (MM:SS)"""
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed//60:02d}:{elapsed%60:02d}"

    def update(self, label):
        """
        라벨 업데이트 및 카운팅 로직
        - 쿨다운 시스템으로 중복 카운트 방지
        - 운동 상태 전환 감지 (Stand -> 운동 -> Stand 패턴만 카운트)
        """
        self.frame_count += 1
        self.label_counts[label] += 1
        
        # 카운트 쿨다운 감소
        if self.counting_cooldown > 0:
            self.counting_cooldown -= 1
        
        is_exercising = label in [1, 2, 3, 4]  # 0번(시작자세) 제외
        is_standing = label == 0  # 시작자세(서있는 상태)
        
        # 카운트 조건 강화: Stand 상태에서 운동 상태로 전환할 때만 카운트
        should_count = False
        if (is_exercising and self.prev_exercising == False and 
            label != 0 and self.counting_cooldown == 0):
            
            # 추가 조건: 최근에 Stand 상태가 있었는지 확인
            recent_stand = False
            if hasattr(self, 'recent_predictions') and len(self.recent_predictions) > 0:
                recent_stand = 0 in self.recent_predictions[-3:]  # 최근 3개 중 Stand가 있었는지
            
            # Stand -> 운동 전환일 때만 카운트
            if recent_stand or not hasattr(self, 'recent_predictions'):
                should_count = True
                self.count += 1
                self.exercise_labels.append(label)
                self.counting_cooldown = 10  # 10프레임 쿨다운
                
                # 점수 계산
                if label == 1:          # 정상자세
                    self.perfect += 1
                    self.score_log.append(1.0)
                elif label in [2, 3]:   # 단일 문제
                    self.score_log.append(0.5)
                elif label == 4:        # 복합 문제
                    self.score_log.append(0.0)
        
        self.prev_exercising = is_exercising
        
        # 피드백 메시지 생성
        feedback_map = {
            0: "시작 자세를 유지하세요",
            1: "완벽한 자세입니다!",
            2: "무릎을 더 넓게 벌려주세요",
            3: "허리를 곧게 펴고 엉덩이를 뒤로",
            4: "자세를 전체적으로 교정하세요"
        }
        feedback = feedback_map.get(label, "자세 확인 필요")
        
        return feedback, should_count, self.count >= self.max_per_set

    def reset_set(self):
        """세트 완료 후 초기화"""
        self.count = 0
        self.perfect = 0
        self.prev_exercising = False
        self.counting_cooldown = 0

    def save_results(self):
        """
        운동 결과 CSV 저장
        - 인코딩 문제 해결 (UTF-8 BOM)
        - 파일 잠김 시 백업 파일 생성
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 실제 운동 통계 계산
        total_reps = len(self.score_log)
        perfect_reps = sum(1 for s in self.score_log if s == 1.0)
        actual_knee = sum(1 for label in self.exercise_labels if label == 2)
        actual_hip = sum(1 for label in self.exercise_labels if label == 3)
        actual_complex = sum(1 for label in self.exercise_labels if label == 4)
        
        # 점수 계산
        score = min(100, (sum(self.score_log) / len(self.score_log)) * 100) if total_reps > 0 else 0
        
        # 평가 생성
        if score >= 90:
            feedback, comment = "Excellent", "Perfect form! Keep it up!"
        elif score >= 75:
            feedback, comment = "Good", "Great form, minor improvements needed"
        elif score >= 60:
            feedback, comment = "Fair", "Focus on knee and hip position"
        elif score > 0:
            feedback, comment = "Poor", "Practice basic posture slowly"
        else:
            feedback, comment = "No Exercise", "Workout not completed"
        
        # CSV 저장 (개선된 오류 처리)
        csv_path = os.path.abspath("squat_scores.csv")
        
        try:
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                
                # 헤더 작성
                if not file_exists:
                    writer.writerow([
                        "timestamp", "user_id", "time", "sets", "total", 
                        "perfect", "knee", "hip", "complex", "score", "feedback", "comment"
                    ])
                
                # 데이터 작성
                writer.writerow([
                    timestamp, self.user_id, self.get_time(), self.set_count, 
                    total_reps, perfect_reps, actual_knee, actual_hip, actual_complex, 
                    f"{score:.1f}", feedback, comment
                ])
                
            print(f"저장 완료: {csv_path}")
            
        except PermissionError:
            # 백업 파일로 저장
            backup_path = csv_path.replace('.csv', f'_backup_{int(time.time())}.csv')
            print(f"백업 파일로 저장: {backup_path}")
            
            with open(backup_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "user_id", "time", "sets", "total", 
                    "perfect", "knee", "hip", "complex", "score", "feedback", "comment"
                ])
                writer.writerow([
                    timestamp, self.user_id, self.get_time(), self.set_count, 
                    total_reps, perfect_reps, actual_knee, actual_hip, actual_complex, 
                    f"{score:.1f}", feedback, comment
                ])
            
        except Exception as e:
            print(f"저장 실패: {e}")
        
        # 결과 출력
        print(f"\n최종 점수: {score:.1f}점")
        print(f"총 운동: {total_reps}회, 완벽: {perfect_reps}회")
        print(f"무릎문제: {actual_knee}회, 골반문제: {actual_hip}회, 복합문제: {actual_complex}회")
        print(f"평가: {feedback} - {comment}")

def run():
    """
    메인 실행 함수
    - 웹캠 캡처 및 실시간 포즈 분석
    - UI 표시 및 사용자 피드백
    """
    print("Posecoach 스쿼트 분석기 시작")
    print("모델 로딩 중...")
    
    # 모델 및 스케일러 로딩
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("모델 로딩 완료")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return

    # 라벨 정의
    labels = ["시작자세", "정상자세", "무릎문제", "골반문제", "복합문제"]
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 열기 실패")
        return

    # 창 설정
    window_name = "Posecoach - Squat Analyzer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 세션 시작
    session = SquatSession()
    speak_async("포즈코치 스쿼트 분석을 시작합니다")
    
    frame_count = 0
    landmarks = None
    texts = ['감지 중...', '', '0/10', '0', '00:00']

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break
                
            frame_count += 1

            # 3프레임마다 분석 (성능 최적화 - 5에서 3으로 변경)
            if frame_count % 3 == 0:
                feature, landmarks = extract_pose_features(frame)
                
                if feature is not None:
                    try:
                        # 모델 예측
                        pred_proba = model.predict(scaler.transform(feature), verbose=0)
                        pred_class = int(np.argmax(pred_proba))
                        confidence = float(np.max(pred_proba))
                        
                        # 예측 안정화 (상체만 보일 때 Stand로 보정)
                        if confidence < 0.7:
                            pred_class = 0  # 신뢰도 낮으면 Stand 상태로
                        
                        # 최근 예측 버퍼 관리
                        if not hasattr(session, 'recent_predictions'):
                            session.recent_predictions = []
                        
                        session.recent_predictions.append(pred_class)
                        if len(session.recent_predictions) > 3:  # 버퍼 크기 축소 (5->3)
                            session.recent_predictions.pop(0)
                        
                        # 다수결 방식으로 최종 라벨 결정 (Stand 우선)
                        if len(session.recent_predictions) >= 2:
                            most_common = Counter(session.recent_predictions).most_common(1)[0]
                            if most_common[1] >= 2:
                                pred_class = most_common[0]
                            # Stand가 포함되어 있으면 우선 적용
                            elif 0 in session.recent_predictions:
                                pred_class = 0
                        
                        # 세션 업데이트
                        feedback, should_count, set_done = session.update(pred_class)
                        
                        # UI 텍스트 업데이트
                        cooldown_text = f" (쿨다운:{session.counting_cooldown})" if session.counting_cooldown > 0 else ""
                        texts = [
                            f"자세: {labels[pred_class]}{cooldown_text}",
                            f"피드백: {feedback}",
                            f"카운트: {session.count}/{session.max_per_set}",
                            f"세트: {session.set_count}",
                            f"시간: {session.get_time()}"
                        ]
                        
                        # 카운트 이벤트 처리
                        if should_count:
                            print(f"{session.count}회 완료! - {labels[pred_class]}")
                            
                            # 음성 피드백
                            if session.count <= 9:
                                speak_async(str(session.count))
                            elif session.count == 10:
                                speak_async("1세트 완료! 잠시 휴식하세요")
                            
                            # 잘못된 자세 시 경고음
                            if pred_class in [2, 3, 4]:
                                play_beep()
                        
                        # 세트 완료 처리
                        if set_done:
                            session.set_count += 1
                            session.reset_set()
                            print(f"{session.set_count}세트 완료!")
                            time.sleep(3)  # 휴식 시간
                            speak_async("새로운 세트를 시작하세요")
                            
                    except Exception as e:
                        print(f"예측 오류: {e}")
                        texts[0] = "예측 오류"
                else:
                    texts[0] = "포즈 감지 실패 - 카메라에서 멀어지세요"

            # 프레임 시각화
            if landmarks:
                frame = draw_landmarks_on_frame(frame, landmarks)

            # 텍스트 오버레이
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 1 and "완벽" in text else (0, 255, 255)
                frame = simple_text(frame, text, (10, 30 + i * 40), color)

            # FPS 표시
            fps_text = f"FPS: {cv2.getTickFrequency() / (cv2.getTickCount() - frame_count * cv2.getTickFrequency() / 30):.1f}"
            cv2.putText(frame, fps_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("종료 중...")
                speak_async("오늘도 수고하셨습니다!")
                time.sleep(2)
                session.save_results()
                break
            elif key == ord('r'):  # 리셋 기능 추가
                session = SquatSession()
                speak_async("세션이 리셋되었습니다")

    except KeyboardInterrupt:
        print("사용자 중단")
    except Exception as e:
        print(f"실행 오류: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("프로그램 종료")

if __name__ == "__main__":
    run()