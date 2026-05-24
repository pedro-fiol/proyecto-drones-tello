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
FB_MAX_VELOCITY_CM_S = 100
LR_MAX_VELOCITY_CM_S = 60

MANUAL_FB_VELOCITY_CM_S    = 80
MANUAL_LR_VELOCITY_CM_S    = 80
MANUAL_UD_VELOCITY_CM_S    = 80
MANUAL_YAW_VELOCITY_DEG_S  = 80

# ---- Target tracking ----
TARGET_ALTITUDE_CM       = 150
MAX_TRACKING_ALTITUDE_CM = 180   # ceiling clamp 
MIN_TRACKING_ALTITUDE_CM = 30   # floor clamp 

TARGET_LOST_GRACE_S = 4.0 # to avoid altitude PID flickering; longer = more time yawing toward last-seen side before full search FSM

INTERCEPT_DISTANCE_CM       = 95
FRONT_TOF_MAX_RANGE_CM      = 120
FRONT_TOF_WALL_STOP_CM      = 60

# Diagonal-wall mitigation: when front ToF goes from valid reading → -1 (OOR) in one
# poll, drone likely just crossed past a wall edge (narrow ToF cone now looking past wall).
# Freeze forward motion this many seconds to avoid crashing into the wall diagonally.
FRONT_TOF_DISCONTINUITY_FREEZE_S = 0.5

# Pitch source hysteresis: stay in ToF mode for N consecutive invalid frames before
# switching to bbox fallback. Stops ToF↔bbox flicker at edge of ToF range.
FRONT_TOF_INVALID_HYSTERESIS_FRAMES = 3

# Detection hysteresis: keep treating target as visible for N consecutive miss frames.
# YOLO confidence dips below threshold for a single frame trigger grace/EMA reset →
# yaw PID restarts on every dropout → constant oscillation. Hold last smoothed state instead.
TRACKING_MISS_HYSTERESIS_FRAMES = 45


# --- Search algorithm ---
# Tello has no reliable XY odometry → no dead-reckoning. Advance ends via pitch_pid
# settling on INTERCEPT_DISTANCE_CM. Budget tracked in cycles, not cm.
SEARCH_SPIN_VELOCITY_DEG_S    = 50
SEARCH_YAW_TOLERANCE_DEG      = 5
SEARCH_SAMPLE_EVERY_DEG       = 10       # min yaw delta between direction samples during spin
SEARCH_MAX_CYCLES             = 3        # spin+advance cycles before hover_done
SEARCH_ADVANCE_TOLERANCE_CM   = 15       # advance done when |INTERCEPT_DISTANCE_CM - front_tof_cm| < this
SEARCH_ADVANCE_TIMEOUT_S      = 8.0      # bail if PID never settles (open space, ToF dropouts)
SEARCH_OPEN_SPACE_VELOCITY_CM_S = 30     # fb when ToF out-of-range during advance (no obstacle yet → push forward until ToF acquires)

# Advance distance cap: integrate fb*dt during advance. Force re-scan after this many cm.
# Limits diagonal-wall crash damage range — drone never blindly pushes more than this without re-scanning.
SEARCH_ADVANCE_MAX_DISTANCE_CM = 100

# Mid-advance yaw sweep: every N seconds during advance, pause fb and do a ±arc sweep
# scanning for off-axis walls. If ToF acquires anything <= SWEEP_WALL_CM, abort advance → re-spin.
SEARCH_ADVANCE_SWEEP_EVERY_S    = 2.0
SEARCH_ADVANCE_SWEEP_YAW_DEG_S  = 50    # yaw rate during sweep (gentler than spin)
SEARCH_ADVANCE_SWEEP_ARC_DEG    = 20    # sweep amplitude ± around current heading
SEARCH_ADVANCE_SWEEP_WALL_CM    = 80    # any ToF reading <= this during sweep = diagonal wall detected

# Grace-period yaw recovery: when target lost off-edge, yaw toward last-seen side
GRACE_RECOVERY_YAW_DEG_S      = 60       # yaw rate during grace toward last-seen side
GRACE_RECOVERY_MIN_OFFSET_PX  = 100      # only trigger if target was that far off-center when lost



# ---- PID gains (kp, ki, kd) ----

GAINS_ALTITUDE_PID          = (1.4, 0.04, 0.08)         # cm error  → ud  (baro mode)
GAINS_ALTITUDE_TARGET_PID   = (-0.15, -0.004, -0.08)    # px error  → ud  (image-y inverted vs world-up)
GAINS_YAW_PID               = (0.2, 0.003, 0.1)       # px error  → yaw
GAINS_FORWARD_BACK_TOF_PID = (-0.85, -0.02, -0.42)   # cm error → fb
GAINS_FORWARD_BACK_BBOX_PID = (200, 0.0, 20.0)   # ratio err → fb
GAINS_LEFT_RIGHT_PID        = (0.2, 0.01, 0.15)   # px error  → lr

"""
GAINS_ALTITUDE_PID          = (0, 0.04, 0.08)         # cm error  → ud  (baro mode)
GAINS_ALTITUDE_TARGET_PID   = (-0.15, -0.004, -0.08)    # px error  → ud  (image-y inverted vs world-up)
GAINS_YAW_PID               = (0, 0, 0)       # px error  → yaw
GAINS_FORWARD_BACK_TOF_PID = (0, 0, 0)   # cm error → fb
GAINS_FORWARD_BACK_BBOX_PID = (0, 0.0, 0.0)   # ratio err → fb
GAINS_LEFT_RIGHT_PID        = (0.2, 0.01, 0.15)   # px error  → lr
"""

# ---- Failure tiers ----
BATTERY_MEDIUM_PERCENT      = 5
BATTERY_HARD_PERCENT        = 3
CONNECTION_DROP_MEDIUM_S    = 5
CONNECTION_DROP_HARD_S      = 5
TOF_STUCK_MEDIUM_S          = 5
ERROR_HOVER_TIMEOUT_S       = 5

# ---- RC loop ----
RC_LOOP_INTERVAL_S          = 0.05   # 20 Hz

# ---- Webapp manual control ----
# Heartbeat grace for the browser dpad. Each /cmd/manual_set call refreshes
# the deadline; if the browser tab dies or network drops, manual_* zero within
# this many seconds and the drone holds position (Tello baro + optical-flow).
# Long enough to absorb a missed 100 ms heartbeat, short enough that runaway
# motion stops fast.
WEB_MANUAL_HEARTBEAT_GRACE_S = 0.6


# ---- ReID (Phase 8: multi-target lock) ----
# Backbone + weights. OSNet x0_25 trained on MSMT17 — small (~3MB), fast (~3ms/crop GPU),
# and trained on the most varied person-ReID dataset available so it generalizes to
# indoor drone footage. Weights file is downloaded on first run via gdown.
REID_MODEL_NAME            = "osnet_x0_25"
REID_WEIGHTS_PATH          = "models/osnet_x0_25_msmt17.pt"
REID_WEIGHTS_GDRIVE_ID     = "1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e"  # osnet_x0_25_msmt17 (torchreid model zoo)
REID_INPUT_SIZE_HW         = (256, 128)             # ReID standard input
REID_PIXEL_MEAN            = (0.485, 0.456, 0.406)  # ImageNet normalization
REID_PIXEL_STD             = (0.229, 0.224, 0.225)

# Lock match: cosine distance threshold. Below = same person, above = different.
# 0.30 is a sane default for OSNet on MSMT17 — typical same-person dist 0.05-0.20,
# typical cross-person dist 0.35-0.70. Tune via lock_response.csv log if needed.
LOCK_MATCH_THRESHOLD       = 0.30

# Lock-time EMA on locked embedding. Each frame the locked target is matched,
# embedding moves alpha toward the new sample. Absorbs slow pose/lighting drift
# without losing identity. alpha=0.05 → ~20-frame time constant.
LOCK_EMBEDDING_EMA_ALPHA   = 0.05


# ---- Target tracking (Phase 6a) ----
# Closeness setpoints used by the pitch fallback PID when front ToF is invalid.
# Each Target picks one of these via target.closeness_setpoint. Higher closeness =
# closer person. Drone advances under the bbox PID until closeness reaches setpoint.
#
# BBOX_HEIGHT_RATIO_SETPOINT — for nose/eyes/bbox targets that frame the full body.
#   0.75 ≈ bbox fills 75% of frame height ≈ ~50cm distance. Pushes the drone INTO
#   the ToF acquisition range when ToF is blind (cone misalignment).
#
# BBOX_WIDTH_RATIO_SETPOINT — for SHOULDERS_MIDPOINT_TARGET. Person bbox width normalized
#   by frame width. More reliable than shoulder keypoint spread (works even if one
#   shoulder keypoint is low-confidence). Starts at 0.35 — tune by walking to known
#   distance (e.g. 80cm from drone) and reading closeness from HUD/log.
#
# SHOULDER_WIDTH_RATIO_SETPOINT — kept for reference / fallback.
#
# The active TARGET selection lives in interceptor/target.py to avoid circular imports.
BBOX_HEIGHT_RATIO_SETPOINT     = 1
BBOX_WIDTH_RATIO_SETPOINT      = 0.45
SHOULDER_WIDTH_RATIO_SETPOINT  = 0.35



