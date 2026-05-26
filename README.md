# Tello Interceptor

Sistema autónomo de **persecución e interceptación de personas en interiores** usando un **DJI RoboMaster Tello Talent (RMTT)**.

El dron detecta personas con YOLOv8-pose, selecciona un objetivo (por identidad ReID o por bbox más grande), y lo persigue cerrando el lazo en los cuatro ejes (altitud, yaw, avance y desplazamiento lateral) mediante controladores PID independientes. Cuando pierde al objetivo entra en una FSM de búsqueda (spin + advance + sweep) hasta reencontrarlo. Toda la operación se monitoriza en tiempo real desde una consola web (FastAPI + WebSocket) que también permite tomar control manual.

> 🎥 **[DEMO PRINCIPAL — vídeo aquí]**
> *Coloca un vídeo de ~30–60 s mostrando: detección → lock por click → seguimiento autónomo → pérdida + búsqueda → reenganche.*

---

## Índice

1. [Hardware](#hardware)
2. [Instalación rápida](#instalación-rápida)
3. [Arquitectura](#arquitectura)
4. [Pipeline de percepción](#pipeline-de-percepción)
5. [Multi-target lock (ReID)](#multi-target-lock-reid)
6. [Sistema de targets](#sistema-de-targets)
7. [Control PID (4 ejes)](#control-pid-4-ejes)
8. [Máquina de estados del vuelo](#máquina-de-estados-del-vuelo)
9. [FSM de búsqueda](#fsm-de-búsqueda)
10. [Capas de seguridad](#capas-de-seguridad)
11. [Consola Web (FastAPI)](#consola-web-fastapi)
12. [HUD OpenCV](#hud-opencv)
13. [Modo Phantom](#modo-phantom)
14. [Logs y tuning de PID](#logs-y-tuning-de-pid)
15. [Tests](#tests)
16. [Variables de entorno](#variables-de-entorno)
17. [Atajos de teclado](#atajos-de-teclado)
18. [Estructura del repositorio](#estructura-del-repositorio)

---

## Hardware

| Componente | Detalle |
|---|---|
| Dron | DJI RoboMaster Tello Talent (RMTT), SSID `RMTT-AD3294` |
| Expansión | RmTTOC ESP32 — ToF frontal VL53Lx + matriz de puntos |
| IP del dron | `192.168.10.1` (modo AP, fijo) |
| Stream de vídeo | 960 × 720 @ ~30 FPS |
| Sensores propios | Barómetro, ToF inferior, IMU (pitch / roll / yaw / aceleraciones) |
| Sensor añadido | ToF frontal vía `EXT tof?` (capital, sensible a mayúsculas) |

> 📷 **[Foto del dron equipado — `fototello1.jpeg` / `fototello2.jpeg` ya en repo]**
> *Insertar las dos imágenes en miniatura una al lado de otra.*

---

## Instalación rápida

> Requisitos: Windows, **Python 3.12** (CUDA-enabled PyTorch), red WiFi.

```powershell
# 1. Crear venv y dependencias
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # ultralytics, djitellopy, fastapi,
                                                # uvicorn, torchreid, opencv-python, ...

# 2. Conectar PC a la red WiFi del dron (cambia el adaptador automáticamente)
powershell -ExecutionPolicy Bypass -File switch-tello.ps1 tello

# 3. Lanzar
.venv\Scripts\python.exe main.py
```

Para volver a la red de área local:

```powershell
powershell -ExecutionPolicy Bypass -File switch-tello.ps1 lan
```

La consola web queda accesible en `http://localhost:8000` (o la IP del PC en la red del dron).

---

## Arquitectura

```
                       ┌────────────────────────────────────────┐
                       │              main.py                   │
                       │  carga .env, instancia interceptor,    │
                       │  lanza webapp en thread daemon         │
                       └──────────────────┬─────────────────────┘
                                          │
                       ┌──────────────────▼─────────────────────┐
                       │      TelloInterceptor (orquestador)    │
                       │  · _video_loop  (main thread, ~30 FPS) │
                       │  · _rc_control_loop (20 Hz)            │
                       │  · _update_front_tof (poll EXT tof?)   │
                       │  · _update_telemetry (state listener)  │
                       └─┬──────┬──────┬──────┬──────┬──────┬───┘
                         │      │      │      │      │      │
                ┌────────┘      │      │      │      │      └────────┐
                ▼               ▼      ▼      ▼      ▼               ▼
          perception.py   target.py reid.py PID hud_overlay  webapp/server.py
          YOLOv8-pose     Target+   OSNet  4 ejes  OpenCV    FastAPI + WS + MJPEG
                          LockState x0_25
```

> 📊 **[Diagrama de flujo detallado]**
> *Renderizar `flux_diagram.md` (Obsidian Canvas) como PNG/SVG e insertar aquí. Cubre los 10 grupos: ENTRY · ORCHESTRATOR · THREADS · PERCEPTION · MULTI-TARGET · CONTROL · SEARCH FSM · SAFETY · WEBAPP · SHUTDOWN.*

### Módulos (paquete `interceptor/`)

| Archivo | Función |
|---|---|
| `tello_interceptor.py` | Orquestador. Posee el `Tello`, el `_sdk_lock` y todos los threads |
| `constants.py` | **Única fuente** de constantes: ganancias, límites, gates, umbrales |
| `perception.py` | YOLOv8-pose → `list[Person]` con bbox + keypoints (nariz, ojos, orejas, hombros) |
| `target.py` | Abstracción `Target` (predicado de visibilidad + extractor de punto + closeness) + `LockState` |
| `reid.py` | `ReidEmbedder` (OSNet x0_25 / MSMT17) → embeddings 512-D L2-normalizados |
| `pid_controller.py` | Controlador PID genérico con clamp anti-windup y salida saturada |
| `hud_overlay.py` | Dibujado sobre frame: bboxes, keypoints, crosshair, telemetría, badges |

Otros:

- `webapp/server.py` — servidor FastAPI (HTTP + WebSocket + MJPEG)
- `webapp/static/index.html` — consola operador (video + dpad + lock + telemetría)
- `tests/` — `pytest` para perception, PID y target (sin dron)

---

## Pipeline de percepción

Cada frame del stream:

1. **YOLO** (`yolov8s-pose.pt`, conf ≥ 0.8, clase `person`) → lista de `Person`.
2. Cada `Person` lleva bbox + 7 keypoints COCO seleccionados (nariz, ojos, orejas, hombros).
3. **Embedding ReID** opcional: si `models/osnet_x0_25_msmt17.pt` está disponible, cada bbox se recorta y se pasa por OSNet en batch (≈ 3 ms / persona en CUDA) generando un vector 512-D L2-normalizado.
4. **Selección del objetivo activo** (`select_target_person`):
   - **Sin lock** → persona con bbox mayor, con prior espacial de "cercana a la última posición suavizada".
   - **Con lock** → distancia coseno mínima frente al embedding bloqueado. Si pasa de `LOCK_MATCH_THRESHOLD = 0.30` se considera pérdida.
5. **EMA** sobre la posición del punto (`alpha = 0.3`, ~5 frames) y sobre el closeness (mismo alpha). Mata el jitter ±5 px del detector antes de derivar.
6. **Histéresis de detección** (`TRACKING_MISS_HYSTERESIS_FRAMES = 45`) — caídas momentáneas de confianza no abren el FSM de búsqueda; se mantiene el último estado suavizado.

> 🎥 **[Demo: detección + keypoints]**
> *Clip mostrando bbox + esqueleto + crosshair sobre el operador moviéndose.*

---

## Multi-target lock (ReID)

Cuando hay **varias personas** en el frame, el dron debe perseguir a una concreta y no saltar de una a otra. Solución: identidad por **embedding facial-corporal**, no por track ID.

- Modelo: **OSNet x0_25** entrenado en **MSMT17** (`models/osnet_x0_25_msmt17.pt`, ~9 MB).
- Cada detección se convierte en un vector 512-D unitario.
- `LockState.embedding` guarda la huella del objetivo escogido.
- En cada frame: `dist = 1 − cos(embedding_lock, embedding_i)`. La detección con menor distancia que pase el umbral es el objetivo.
- **EMA del embedding bloqueado** (`alpha = 0.05`, ~20 frames) absorbe deriva lenta (pose, iluminación) sin perder identidad.
- Si ninguna detección pasa el umbral → entra en `hover` (grace 4 s) → `searching`.

**Tres formas de adquirir el lock:**

| Acción | Resultado |
|---|---|
| Tecla `I` / botón "Lock" | Desbloqueado → bloquea el bbox más grande. Bloqueado → cicla left-to-right entre detecciones |
| Click sobre el vídeo (web) | Bloquea sobre la persona cuyo bbox contiene el punto cliqueado |
| Tecla `C` / botón "Clear" | Libera el lock → vuelve a "bbox más grande" |

> 🎥 **[Demo crítica: lock multi-target]**
> *Escena con 2–3 personas. Mostrar (a) click-to-lock sobre una, (b) las otras se cruzan delante, (c) el dron sigue a la persona correcta. Esta es la feature estrella de la rama `develop-multiple-targets`.*

---

## Sistema de targets

`Target` es un dataclass inmutable que describe **qué punto del cuerpo sigue el dron y dónde lo encuadra**. Cuatro vienen predefinidos:

| Target | Punto seguido | Encuadre Y | Señal de cercanía | Setpoint |
|---|---|---|---|---|
| `nose` | Centro de la nariz | 0.20 (alto) | bbox height / frame height | 1.00 |
| `eyes_midpoint` | Punto medio de ambos ojos | 0.20 | bbox height / frame height | 1.00 |
| `shoulders_midpoint` (default) | Centro de los hombros | 0.25 | bbox width / frame width | 0.45 |
| `bbox_center` | Centro del bbox completo | 0.50 | bbox height / frame height | 1.00 |

Cada `Target` aporta:

- `is_visible(person)` — gate de keypoints requeridos.
- `point(person)` — devuelve `(x, y)` que alimenta los PID de yaw / altitud / roll.
- `closeness(person)` y `closeness_setpoint` — usados por el **PID de pitch en modo bbox** cuando el ToF frontal no es válido.

El target es **intercambiable en runtime** desde la consola web (dropdown `target_name`). Al cambiar se resetean integrales y EMAs para no propagar estados de un target a otro.

> 🎥 **[Demo: cambio de target]**
> *Cambiar `shoulders_midpoint` → `nose` desde la web y observar cómo el dron se eleva para reencuadrar.*

---

## Control PID (4 ejes)

Cuatro lazos PID corren en paralelo a **20 Hz** (`RC_LOOP_INTERVAL_S = 0.05 s`):

| Eje | Variable controlada | Error | Ganancias `(Kp, Ki, Kd)` | Salida |
|---|---|---|---|---|
| **Altitud (baro)** | `height_cm` | `target_alt − height_cm` (cm) | `(1.4, 0.04, 0.08)` | `ud` cm/s (clamp ±60) |
| **Altitud (target)** | `target_y_px` | `target_y_smoothed − y_setpoint` (px) | `(−0.15, −0.004, −0.08)` | `ud` cm/s (clamp ±60) |
| **Yaw** | `target_x_px` | `target_x_smoothed − frame_center_x` (px) | `(0.2, 0.003, 0.1)` | `yaw` deg/s (clamp ±50) |
| **Pitch (ToF)** | distancia frontal | `intercept_dist − front_tof_cm` (cm) | `(−0.85, −0.02, −0.42)` | `fb` cm/s (clamp ±100) |
| **Pitch (bbox)** | closeness | `setpoint − closeness_smoothed` | `(200, 0, 20)` | `fb` cm/s (clamp ±100) |
| **Roll** | `target_x_px` (fuera de bbox) | `target_x − frame_center_x` (px) | `(0.2, 0.01, 0.15)` | `lr` cm/s (clamp ±60) |

**Convención de signo del error**: la altitud usa `target − medida` (imagen-y invertida vs. mundo arriba), el yaw usa `medida − target`. Es **intencional** — no unificar sin verificar ambos ejes a la vez.

**Detalles importantes del lazo:**

- **Anti-windup** por saturación + reset explícito (`reset_integral()`) al cambiar de modo, target o tipo de fuente de pitch.
- **Hand-off ToF ↔ bbox** en el eje de pitch: histéresis `FRONT_TOF_INVALID_HYSTERESIS_FRAMES = 3` evita parpadeo en el borde de los 120 cm. Al cambiar de fuente, el `error_last` del PID entrante se siembra con el error actual para que el término D no salte en la primera muestra (las unidades cambian de cm a ratio).
- **Roll con dead-zone de bbox**: si el centro horizontal de la imagen cae dentro del bbox suavizado del objetivo, `lr = 0`. Evita oscilaciones laterales innecesarias cuando el target ya está "delante".

> 📈 **[Plots de respuesta PID]**
> *Generar PNGs desde `logs/*.csv` con `plot_altitude.py`, `plot_yaw.py`, `plot_pitch.py` e incrustar una pequeña galería: altitud / yaw / pitch / roll. Estos son los gráficos de tuning experimentales.*

---

## Máquina de estados del vuelo

El video loop selecciona un modo cada tick según `is_airborne`, `tracking_target`, y el grace timer:

```
            ┌─────────┐  takeoff (SPACE)   ┌──────────────┐
            │ grounded│ ──────────────────▶│ intercepting │◀──┐
            └─────────┘                    └──────┬───────┘   │
                  ▲                               │           │ target
                  │ land (SPACE)                  │ target    │ reacquired
                  │                               │ lost      │
                  │                          ┌────▼─────┐     │
                  │                          │  hover   │─────┘
                  │                          └────┬─────┘
                  │                               │ > TARGET_LOST_GRACE_S (4 s)
                  │                          ┌────▼─────┐
                  │                          │searching │ ── FSM de búsqueda
                  │                          └──────────┘
                  │
                  └── manual (M) ◀──▶ cualquier modo (override de RC)
```

- **grounded** — sin RC, todos los integrales en cero.
- **intercepting** — todos los PID activos (altitud-target, yaw, pitch, roll).
- **hover** — `ud = 0` (Tello mantiene con baro + optical-flow). El yaw apunta hacia el último lado conocido si el target se perdió por borde.
- **searching** — altitud-baro activa, FSM de búsqueda controla yaw/fb.
- **manual** — `lr/fb/ud/yaw` provienen del teclado o del dpad web (con heartbeat 0.6 s).

---

## FSM de búsqueda

Cuando el grace de 4 s expira sin reencontrar al objetivo, el dron entra en `searching`. **No hay odometría XY fiable** en el Tello, así que la estrategia es reactiva:

```
        ┌─────────┐
   ──▶  │  spin   │  yawea a SEARCH_SPIN_VELOCITY_DEG_S (50°/s)
        └────┬────┘  hacia el último lado conocido del target.
             │       Acumula ángulo hasta 360° → garantiza barrido completo.
             │
             ▼
        ┌─────────┐  PID de pitch sobre ToF frontal con setpoint
   ┌──▶ │ advance │  INTERCEPT_DISTANCE_CM (95 cm). Si ToF = OOR
   │    │ moving  │  empuja a SEARCH_OPEN_SPACE_VELOCITY_CM_S.
   │    └────┬────┘
   │         │ cada SEARCH_ADVANCE_SWEEP_EVERY_S (2 s)
   │         ▼
   │    ┌─────────┐  fb = 0, yaw ± SWEEP_ARC_DEG (20°). Si durante el
   │    │ advance │  barrido ToF lee ≤ 80 cm → muro en diagonal,
   │    │ sweeping│  aborta el advance y vuelve a spin.
   │    └────┬────┘
   │         │
   └─────────┘  Bucle hasta:
                · ToF | error | < 15 cm (objetivo a distancia de interceptación)
                · timeout 8 s
                · distancia integrada > 100 cm (cap por seguridad)
                → vuelve a spin
```

Si en cualquier momento se reencuentra al objetivo (`tracking_target == True`), el FSM se resetea y se vuelve a `intercepting`.

---

## Capas de seguridad

Antes de cada envío de RC se aplican gates **acumulables**:

1. **Geofence vertical** — bloquea `ud > 0` si `height_cm ≥ MAX_TRACKING_ALTITUDE_CM` (180 cm) y `ud < 0` si `≤ MIN_TRACKING_ALTITUDE_CM` (30 cm).
2. **Wall-stop** — si `front_tof_cm ≤ FRONT_TOF_WALL_STOP_CM` (60 cm) y `fb > 0`, fuerza `fb = 0` y resetea integrales de pitch.
3. **Mitigación de pared diagonal** — al detectar transición `front_tof_cm: válido → −1` (cono ToF cruzando un borde) se congela `fb` durante `FRONT_TOF_DISCONTINUITY_FREEZE_S = 0.5 s`.
4. **Heartbeat de mando web** — si el navegador deja de enviar `manual_set` durante > 0.6 s, las velocidades manuales se ponen a 0 (el dron flotará). Evita drone runaway si la pestaña muere.
5. **Modo phantom** — desactiva por completo el envío de comandos al hardware (sólo HUD).

> 🎥 **[Demo: wall-stop]**
> *Dron avanzando hacia una pared, ToF frontal disparándose a 60 cm, parada en seco visible en el HUD.*

---

## Consola Web (FastAPI)

> 🖼️ **[Captura de la consola web completa]**
> *Mostrar layout: video MJPEG (centro), telemetría (lateral), dpad manual (abajo), botones takeoff / land / lock / clear / target selector.*

Servidor FastAPI lanzado en thread daemon al arrancar `main.py`.

### Endpoints

| Ruta | Tipo | Función |
|---|---|---|
| `GET /` | HTTP | `static/index.html` |
| `GET /video` | MJPEG | Stream multipart del frame anotado (JPEG q=80, ~30 FPS) |
| `WS /ws/telemetry` | WebSocket | Push 20 Hz de todo el estado (RC, sensores, modo, lock, target) |
| `POST /cmd/takeoff_land` | HTTP | Toggle takeoff / land (equiv. SPACE) |
| `POST /cmd/toggle_manual` | HTTP | Entra / sale de manual (equiv. M) |
| `POST /cmd/lock` | HTTP | Lock-cycle (equiv. I) |
| `POST /cmd/clear_lock` | HTTP | Liberar lock (equiv. C) |
| `POST /cmd/lock_at` | HTTP | Click-to-lock con coords normalizadas `{x, y}` ∈ [0, 1] |
| `POST /cmd/set_target` | HTTP | Cambiar target activo (`{name: "nose"|...}`) |
| `POST /cmd/manual_set` | HTTP | Velocidades manuales `{lr, fb, ud, yaw}` (heartbeat) |
| `POST /cmd/stop` | HTTP | Apagado limpio (equiv. ESC) |

### Diseño

- **Sin auth, sin HTTPS** — pensado para `localhost` o LAN sobre el AP del dron. Si se expone a WAN, añadir tunelado y token.
- **Heartbeat manual**: cada `/cmd/manual_set` empuja `_web_manual_until` 0.6 s hacia el futuro. El frontend reenvía cada ~100 ms mientras se pulse un botón del dpad. Pestaña muerta → heartbeat vence → drone para.
- **Telemetría completa** vía WS: `mode, is_airborne, is_manual, battery, front_tof, down_tof, height, pitch, roll, yaw, vx/vy/vz, ax/ay/az, temp, rc[4], locked_track_id, lock_distance, lock_target_idx, num_detections, reid_available, search_state, target_name, available_targets`.

---

## HUD OpenCV

La ventana local (`cv2.imshow("TelloInterceptor")`) muestra el frame anotado con:

- Bboxes + keypoints de todas las detecciones.
- Highlight diferenciado para el objetivo seleccionado y para "objetivo bloqueado por ReID".
- Crosshair central, línea horizontal del setpoint Y del target activo.
- Tira de telemetría: batería, altitudes, ToFs, ángulos, RC actual.
- Badges: `PHANTOM` (si activo), modo de control (`AUTO`/`MANUAL` + valores actuales), warning de pared.
- Etiqueta de estado de detección (target visible, modo actual, nombre del target).

> 🎥 **[Demo: HUD en vivo]**
> *Captura de pantalla o clip corto del HUD mostrando todos los overlays simultáneamente.*

---

## Modo Phantom

Cuando `TELLO_PHANTOM=1` en `.env`:

- El dron **nunca despega** (`takeoff()` skip).
- El RC loop es un **no-op** (no envía nada al hardware).
- Toda la lógica corre: detección, PID, FSM de búsqueda, HUD, web.

Ideal para tunear ganancias y depurar la FSM sin batería ni espacio físico.

```bash
TELLO_PHANTOM=1 .venv\Scripts\python.exe main.py
```

---

## Logs y tuning de PID

Cada axis vuelca un CSV al detener:

| Archivo | Columnas |
|---|---|
| `logs/altitude_response.csv` | `time_s, height_cm, error, ud_command, mode` |
| `logs/yaw_response.csv` | `time_s, target_x, error_x_pixels, yaw_command` |
| `logs/pitch_response.csv` | `time_s, front_tof_cm, closeness, error, fb_command, source` |
| `logs/roll_response.csv` | `time_s, target_x, bbox_left, bbox_right, error_roll, lr_command, inside_bbox` |

Plotters incluidos:

```bash
.venv\Scripts\python.exe plot_altitude.py
.venv\Scripts\python.exe plot_yaw.py
.venv\Scripts\python.exe plot_pitch.py
```

> 📈 **[Capturas de los 4 plots tras una sesión real]**
> *Usar logs existentes en `logs/` para generar PNGs de la respuesta escalón / seguimiento de cada eje.*

---

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

Sin dron. Cubren:

- `test_perception.py` — parsing de keypoints, gates de confianza, propiedades del `Person`.
- `test_pid_controller.py` — compute, saturación, anti-windup, reset.
- `test_target.py` — visibility predicates, point extractors, `select_target_person` con y sin lock.

---

## Variables de entorno

`.env` (gitignored, copiar de `.env.example`):

```bash
TELLO_PHANTOM=0          # 1 = no despegar, solo lógica
TELLO_WEB_HOST=0.0.0.0   # bind del servidor FastAPI
TELLO_WEB_PORT=8000
```

---

## Atajos de teclado

| Tecla | Acción |
|---|---|
| `SPACE` | Takeoff / land toggle |
| `M` | Entrar / salir de modo manual |
| `W / S / A / D` | Forward / back / left / right (manual) |
| `Q / E` | Yaw izquierda / derecha (manual) |
| `↑ / ↓` | Subir / bajar (manual) |
| `I` | Lock-cycle (bbox más grande → ciclar entre detecciones) |
| `C` | Liberar lock |
| `ESC` | Apagado limpio (guarda logs + lands) |

Los mismos comandos están duplicados en la consola web.

---

## Estructura del repositorio

```
proyecto-drones-tello/
├── main.py                    # entry point
├── switch-tello.ps1           # cambio automático WiFi ↔ LAN
├── interceptor/
│   ├── tello_interceptor.py   # orquestador + threads
│   ├── constants.py           # única fuente de constantes
│   ├── perception.py          # YOLOv8-pose → Person
│   ├── target.py              # Target + LockState + selección
│   ├── reid.py                # OSNet embedder
│   ├── pid_controller.py      # PID genérico
│   └── hud_overlay.py         # dibujado sobre frame
├── webapp/
│   ├── server.py              # FastAPI + WS + MJPEG
│   └── static/index.html      # consola operador
├── models/
│   ├── yolov8s-pose.pt        # detector + pose
│   └── osnet_x0_25_msmt17.pt  # ReID (vendored)
├── logs/                      # CSV de respuesta PID
├── tests/                     # pytest, sin dron
├── plot_*.py                  # scripts de visualización
└── flux_diagram.md            # guía del Canvas de Obsidian
```

---

## Limitaciones conocidas

- **Sin odometría XY fiable** en el Tello → la geofence horizontal se desactivó; sólo control vertical estricto.
- **ToF frontal con cono estrecho** → puede "perder" paredes diagonales (mitigado con discontinuity freeze + sweep en advance).
- **YOLOv8s-pose @ 960 × 720** corre ~30 FPS en GPU dedicada; en CPU baja a ~10 FPS y los PID empiezan a sufrir.
- **Lock por ReID** se rompe si la persona cambia drásticamente de pose o sale completamente del frame durante mucho tiempo — el EMA del embedding ayuda pero no es magia.
- **Manejo de varias personas a la vez** requiere ReID disponible (modelo en `models/`); sin él, sólo "biggest bbox".

---

## Licencia y créditos

Proyecto académico — **AEROS, 4º A, UPC, asignatura PD**.
Autor: Pedro FB (`Aszent47`).

Modelos pre-entrenados:
- YOLOv8s-pose — Ultralytics.
- OSNet x0_25 / MSMT17 — KaiyangZhou/deep-person-reid.

Librerías clave: `djitellopy`, `ultralytics`, `torchreid`, `fastapi`, `uvicorn`, `opencv-python`, `pytorch`.
