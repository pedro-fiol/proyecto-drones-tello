"""
interceptor/constants.py
All named constants for the Tello Interceptor project.
No logic here — only numbers and strings with units in names.
"""

# ---- Frame geometry ----
FRAME_WIDTH_PIXELS   = 960
FRAME_HEIGHT_PIXELS  = 720
FRAME_CENTER_X       = FRAME_WIDTH_PIXELS  // 2   # 480
FRAME_CENTER_Y       = FRAME_HEIGHT_PIXELS // 2   # 360

# ---- YOLO / Detection ----
YOLO_MODEL_PATH         = "yolov8s-pose.pt"
YOLO_FRAME_STRIDE       = 1         # detection every N frames
YOLO_PERSON_CLASS_ID    = 0
YOLO_CONFIDENCE_MIN     = 0.7

NOSE_KEYPOINT_INDEX = 0         # COCO keypoint index for nose
NOSE_CONFIDENCE_THRESHOLD = 0.7  # min conf to consider face visible

# ---- Altitude (cm) ----
TARGET_ALTITUDE_CM       = 100
GEOFENCE_MAX_ALTITUDE_CM = 200
GEOFENCE_MIN_ALTITUDE_CM = 20

# ---- Geofence (cm) ----
GEOFENCE_HALF_X_CM             = 200
GEOFENCE_HALF_Y_CM             = 200
GEOFENCE_BACKOFF_VELOCITY_CM_S = 25

# ---- Distance (cm) ----
TRACKING_DISTANCE_CM        = 60
INTERCEPT_DISTANCE_CM       = 40
FRONT_TOF_MAX_RANGE_CM      = 120
FRONT_TOF_WALL_STOP_CM      = 40
FRONT_TOF_WALL_BACKOFF_CM   = 30
WALL_BACKOFF_VELOCITY_CM_S  = 25

# ---- Room / search ----
ROOM_MODE                       = "small"   # "small" | "big"
SEARCH_SPIN_VELOCITY_DEG_S      = 30
SEARCH_ADVANCE_STEP_CM          = 80
SEARCH_MAX_ADVANCE_TOTAL_CM     = 300 # distancia total volada por el dron en una búsqueda


# ---- PID gains (kp, ki, kd) ----
GAINS_ALTITUDE_PID          = (1.5, 0.08, 0.4)   # cm error  → ud
GAINS_YAW_PID               = (0.15, 0.000, 0.25)   # px error  → yaw
GAINS_FORWARD_BACK_TOF_PID  = (0.40, 0.003, 0.20)   # cm error  → fb
GAINS_FORWARD_BACK_BBOX_PID = (90.0, 0.000, 30.0)   # ratio err → fb # solo si front ToF falla.
GAINS_LEFT_RIGHT_PID        = (0.08, 0.000, 0.05)   # px error  → lr

# ---- Failure tiers ----
BATTERY_MEDIUM_PERCENT      = 20
BATTERY_HARD_PERCENT        = 10
CONNECTION_DROP_MEDIUM_S    = 2
CONNECTION_DROP_HARD_S      = 5
TOF_STUCK_MEDIUM_S          = 3
ERROR_HOVER_TIMEOUT_S       = 5

# ---- RC loop ----
RC_LOOP_INTERVAL_S          = 0.05   # 20 Hz

# ---- Mission pad ----
TAKEOFF_PAD_ID = 1   # pad placed at room center, used as XY origin