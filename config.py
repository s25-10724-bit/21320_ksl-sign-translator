# 여러 파일에서 공통으로 사용할 설정값을 모아 둔 파일

from pathlib import Path 

DEFAULT_MAX_FRAMES = 30 # 한 영상에서 사용할 최대 프레임 수

MAX_NUM_HANDS = 2  # MediaPipe가 동시에 인식할 손의 최대 개수(2개)
MIN_DETECTION_CONFIDENCE = 0.5  # 손을 처음 감지할 때 필요한 최소 신뢰도
MIN_TRACKING_CONFIDENCE = 0.5  # 손을 추적할 때 필요한 최소 신뢰도

FEATURES_PER_FRAME = 2 * 21 * 3  # 한 프레임에서 추출되는 특징 개수(양손 2개 × 손당 21개 관절 × x, y, z 좌표 3개 = 126개)
CONFIDENCE_THRESHOLD = 0.65  # 모델이 예측한 확률이 이 값 이상일 때만 결과를 인정
STABLE_COUNT = 8  # 같은 단어가 연속으로 이 횟수 이상 나와야 최종 인식으로 인정

PROJECT_ROOT = Path(__file__).resolve().parent  # 현재 프로젝트 폴더 경로입니다.
