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

# ---- Altitude (cm) ----
ALTITUDE_OFFSET_CM          = 0      
TARGET_ALTITUDE_CM       = 150 - ALTITUDE_OFFSET_CM
GEOFENCE_MAX_ALTITUDE_CM = 180 - ALTITUDE_OFFSET_CM
GEOFENCE_MIN_ALTITUDE_CM = 30 

# Floor for face PID — prevents descent into ground effect / propwash zone.
# Face PID is allowed to climb but blocked from descending below this.
MIN_TRACKING_ALTITUDE_CM = 50

# Grace period after target loss — hover (ud=0) instead of switching to baro PID.
# Stops target/baro PID fight when target flickers for 1-2 frames.
TARGET_LOST_GRACE_S = 1.0

# ud velocity cap (cm/s). Tello SDK accepts ±100, but at full ±100 the rotor response
# is asymmetric → drone tilts → horizontal kick → wall crash. Cap at 40 keeps altitude
# tracking responsive but eliminates pitch coupling.
UD_MAX_VELOCITY_CM_S = 80
YAW_MAX_VELOCITY_CM_S = 50
FB_MAX_VELOCITY_CM_S = 60

# ---- Geofence (cm) ----
GEOFENCE_HALF_X_CM             = 200
GEOFENCE_HALF_Y_CM             = 200
GEOFENCE_BACKOFF_VELOCITY_CM_S = 25

# ---- Distance (cm) ----
TRACKING_DISTANCE_CM        = 80
INTERCEPT_DISTANCE_CM       = 80
FRONT_TOF_MAX_RANGE_CM      = 120
FRONT_TOF_WALL_STOP_CM      = 60
FRONT_TOF_WALL_BACKOFF_CM   = 30
#WALL_BACKOFF_VELOCITY_CM_S  = 5

# ---- Room / search ----
ROOM_MODE                       = "small"   # "small" | "big"
SEARCH_SPIN_VELOCITY_DEG_S      = 30
SEARCH_ADVANCE_STEP_CM          = 80
SEARCH_MAX_ADVANCE_TOTAL_CM     = 300 # distancia total volada por el dron en una búsqueda


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