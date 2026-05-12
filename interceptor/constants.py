"""
interceptor/constants.py
All named constants for the Tello Interceptor project.
"""

# ---- Frame geometry ----
FRAME_WIDTH_PIXELS   = 960
FRAME_HEIGHT_PIXELS  = 720
FRAME_CENTER_X       = FRAME_WIDTH_PIXELS  // 2   # 480
FRAME_CENTER_Y       = FRAME_HEIGHT_PIXELS // 2   # 360

# EMA smoothing on tracked target pixel before PID. YOLO keypoints jitter ±5px @ 30Hz.
# Without smoothing, derivative term spikes ±12 → noisy ud → rotor pitch coupling → drift.
# alpha = new-sample weight. 0.3 = stronger smoothing (~5-frame time constant) to tame Kp=0.35.
TARGET_EMA_ALPHA = 0.3


# ---- YOLO / Detection ----
YOLO_MODEL_PATH         = "yolov8s-pose.pt"
YOLO_FRAME_STRIDE       = 1         # detection every N frames
YOLO_PERSON_CLASS_ID    = 0
YOLO_CONFIDENCE_MIN     = 0.8
NOSE_KEYPOINT_INDEX = 0         # COCO keypoint index for nose
NOSE_CONFIDENCE_THRESHOLD = 0.7  # min conf to consider face visible


# --- Velocity PID controller limits ---
# max is 100
UD_MAX_VELOCITY_CM_S = 60
YAW_MAX_VELOCITY_CM_S = 50
FB_MAX_VELOCITY_CM_S = 60

# ---- Target tracking ----
TARGET_ALTITUDE_CM       = 150
MAX_TRACKING_ALTITUDE_CM = 180   # ceiling clamp 
MIN_TRACKING_ALTITUDE_CM = 30   # floor clamp 

TARGET_LOST_GRACE_S = 1.0 # to avoid altitude PID flickering

INTERCEPT_DISTANCE_CM       = 100
FRONT_TOF_MAX_RANGE_CM      = 120
FRONT_TOF_WALL_STOP_CM      = 40
FRONT_TOF_WALL_BACKOFF_CM   = 20
WALL_BACKOFF_VELOCITY_CM_S  = - FB_MAX_VELOCITY_CM_S  





# ---- PID gains (kp, ki, kd) ----
GAINS_ALTITUDE_PID          = (1.4, 0.04, 0.08)         # cm error  → ud  (baro mode)
GAINS_ALTITUDE_TARGET_PID   = (-0.15, -0.004, -0.08)    # px error  → ud  (image-y inverted vs world-up)
GAINS_YAW_PID               = (0.25, 0.001, 0.05)       # px error  → yaw

GAINS_FORWARD_BACK_TOF_PID = (-1.5, -0.02, -0.4)   # cm error → fb

GAINS_FORWARD_BACK_BBOX_PID = (360.0, 0.00, 5.0)   # ratio err → fb (only when front ToF fails — Kp doubled, Kd cut: ratio signal too noisy for big Kd)
GAINS_LEFT_RIGHT_PID        = (0.08, 0.00, 0.05)   # px error  → lr








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


# ---- Manual keyboard control (per-axis velocities) ----
# Active only when manual_mode toggled via 'M' key. Safety clamps (wall stop,
# altitude clamp) still apply on top — keyboard cannot override safety.
MANUAL_FB_VELOCITY_CM_S    = 30
MANUAL_LR_VELOCITY_CM_S    = 30
MANUAL_UD_VELOCITY_CM_S    = 30
MANUAL_YAW_VELOCITY_DEG_S  = 40

# ---- Target tracking (Phase 6a) ----
# Distance-proxy setpoints used by the pitch fallback PID when front ToF is invalid.
# Each Target picks one of these via target.distance_setpoint. Higher proxy value =
# closer person. Drone advances under the bbox PID until the proxy reaches setpoint.
#
# BBOX_HEIGHT_RATIO_SETPOINT — for nose/eyes/bbox targets that frame the full body.
#   0.75 ≈ bbox fills 75% of frame height ≈ ~50cm distance. Pushes the drone INTO
#   the ToF acquisition range when ToF is blind (cone misalignment).
#
# SHOULDER_WIDTH_RATIO_SETPOINT — for SHOULDERS_MIDPOINT_TARGET. Bbox height is
#   useless when drone hovers chest-high (full body always fills frame). Use the
#   pixel distance between shoulder keypoints divided by frame width instead —
#   it scales meaningfully with distance regardless of framing.
#   0.20 ≈ shoulders ~190px apart on a 960px frame ≈ ~80cm distance.
#
# The active TARGET selection lives in interceptor/target.py to avoid circular imports.
BBOX_HEIGHT_RATIO_SETPOINT     = 0.75
SHOULDER_WIDTH_RATIO_SETPOINT  = 0.20



