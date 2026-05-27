# Tello Interceptor

Dron autónomo de **persecución e interceptación de personas en interiores** sobre un **DJI RoboMaster Tello Talent (RMTT)**.

El dron detecta personas con el modelo YOLOv8-pose, selecciona un objetivo (por identidad ReID o por la caja de detección más grande) y lo persigue con controladores PID para altitud, yaw, roll y pitch. Cuando pierde el objetivo entra en un algoritmo de búsqueda (spin + advance + sweep) hasta reencontrarlo. Toda la operación se monitoriza en tiempo real desde una WebApp (FastAPI + WebSocket) que actúa como dashboard y permite, además, tomar el control manual del dron.

## DEMO PRINCIPAL

Cada escena está grabada desde dos perspectivas (tres en la escena 6):
- **POV 1 — Portátil:** estación de tierra; ejecuta el código y controla el dron desde la WebApp.
- **POV 2 — iPad:** WebApp remota; permite controlar y monitorizar el dron en tiempo real sin ejecutar código.
- **POV 3 — Móvil externo:** solo en la escena 6.

### 1. Controles manuales
<table>
<tr><th>Portátil</th><th>iPad</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/dcaff114-b546-4b66-bd4d-f9c0170b454b" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/998097cf-5cf3-4b5f-b736-208a38d8c1c2" controls width="300"></video></td>
</tr>
</table>

### 2. Tracking sin lock
<table>
<tr><th>Portátil</th><th>iPad</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/9bc79b9b-f7f4-4f5a-9d46-3064b3951138" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/c1d5d563-abe1-4d67-bd7c-b2c5c40aacf2" controls width="300"></video></td>
</tr>
</table>

### 3. Multi-target y cambio de lock
<table>
<tr><th>Portátil</th><th>iPad</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/3dd110c2-425e-4096-b6c3-3d81dfee0ad7" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/85f9ec4a-b941-4864-b7c9-f29d37014223" controls width="300"></video></td>
</tr>
</table>

### 4. Interceptación únicamente del objetivo fijado
<table>
<tr><th>Portátil</th><th>iPad</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/2cf82ba8-78bc-4570-9986-69288b476530" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/1015982a-7b4c-464e-940a-2d44f96eed9f" controls width="300"></video></td>
</tr>
</table>

### 5. Búsqueda dirigida al lock
<table>
<tr><th>Portátil</th><th>iPad</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/bdff4def-3ad4-4c2a-bc73-3526338fc634" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/001994fd-829a-4275-8aa2-e48aa4ed3da4" controls width="300"></video></td>
</tr>
</table>

### 6. Demo del algoritmo de búsqueda completo
<table>
<tr><th>Portátil</th><th>iPad</th><th>Móvil externo</th></tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/30dc0bb2-eb0c-4e20-be47-3ffd23140413" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/617b5047-7af9-4382-a5ea-ef0b23bdc02d" controls width="300"></video></td>
<td><video src="https://github.com/user-attachments/assets/0f5a8ebf-1da0-4083-80f6-d3999bfc441a" controls width="300"></video></td>
</tr>
</table>

---

## Índice

1. [Hardware](#hardware)
2. [Instalación rápida](#instalación-rápida)
3. [Arquitectura](#arquitectura)
4. [Detección](#detección)
5. [Multi-target lock (ReID)](#multi-target-lock-reid)
6. [Sistema de objetivos](#sistema-de-objetivos)
7. [Control PID (4 ejes)](#control-pid-4-ejes)
8. [Máquina de estados del vuelo](#máquina-de-estados-del-vuelo)
9. [Algoritmo de búsqueda](#algoritmo-de-búsqueda)
10. [Capas de seguridad](#capas-de-seguridad)
11. [Consola Web (FastAPI)](#consola-web-fastapi)
12. [HUD OpenCV](#hud-opencv)
13. [Modo Phantom](#modo-phantom)
14. [Atajos de teclado](#atajos-de-teclado)
15. [Estructura del repositorio](#estructura-del-repositorio)
16. [Limitaciones conocidas](#limitaciones-conocidas)

---

## Hardware

| Componente       | Detalle                                                           |
| ---------------- | ----------------------------------------------------------------- |
| Dron             | DJI RoboMaster Tello Talent (RMTT), SSID `RMTT-AD3294`            |
| Kit de expansión | RmTTOC ESP32                                                      |
| IP del dron      | `192.168.10.1`                                                    |
| Stream de vídeo  | 960 × 720 @ ~30 FPS                                               |
| Sensores propios | Barómetro, ToF inferior, IMU (pitch / roll / yaw / aceleraciones) |
| Sensor añadido   | ToF frontal vía `EXT tof?` del kit de expansión                   |

<img width="3024" height="4032" alt="Image" src="https://github.com/user-attachments/assets/06732109-5336-4288-9f8a-d21efd4000fc" />

---

## Instalación rápida

> Requisitos: Windows, portátil con GPU NVIDIA y tarjeta WiFi compatible con redes de 5 GHz, **Python 3.12** (necesario para PyTorch con CUDA), adaptador WiFi externo o un iPhone que comparta datos por cable con el portátil. Cualquier dispositivo conectado al hotspot compartido podrá acceder a la WebApp. No lo he probado en Android, pero existirá una aplicación equivalente a «Dispositivos Apple» que permita compartir datos por cable y evite la necesidad del adaptador WiFi externo.

### 1. Instalar dependencias
```powershell
pip install -r requirements.txt   # ultralytics, djitellopy, fastapi,
                                  # uvicorn, torchreid, opencv-python, ...
```
### 2. Configuración previa
- Acoplar el kit de expansión al dron y encenderlo.
- Conectar un dispositivo móvil al portátil (estación de tierra) por cable para compartir datos; debería aparecer una conexión LAN hacia el hotspot del móvil.
- Conectar el portátil por WiFi al Tello (`RMTT-AD3294`).

<video src="https://github.com/user-attachments/assets/c7073b3f-b14c-4fbe-8f60-f0d0c85bda01" controls width="500"></video>

### 3. Ejecución
```
python main.py
```
La consola web queda accesible en `http://localhost:8000` (`http://0.0.0.0:8000`), o bien en la IP del portátil dentro de la LAN para acceder desde cualquier dispositivo conectado al hotspot.

---

## Arquitectura
Descargar el diagrama de flujo para verlo con más claridad.
![Diagrama de flujo](./Diagrama%20de%20flujo.png)

### `interceptor/`

| Archivo                | Función                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `tello_interceptor.py` | Orquestador. Posee el objeto Tello, el `_sdk_lock` y todos los hilos.                                            |
| `constants.py`         | **Única fuente** de constantes: ganancias, límites y umbrales.                                                   |
| `perception.py`        | YOLOv8-pose → `list[Person]` con bbox y keypoints (nariz, ojos, orejas y hombros).                               |
| `target.py`            | `Target` (visibilidad + extractor de punto + closeness) y `LockState`.                                           |
| `reid.py`              | `ReidEmbedder` (OSNet x0_25 / MSMT17): permite fijar e identificar correctamente a personas distintas.           |
| `pid_controller.py`    | Controlador PID genérico.                                                                                        |
| `hud_overlay.py`       | Dibujado sobre el frame: bboxes, keypoints, crosshair, telemetría e indicadores.                                 |

WebApp:
- `webapp/server.py`: servidor FastAPI (HTTP + WebSocket + MJPEG).
- `webapp/static/index.html`: dashboard (vídeo + dpad + lock + telemetría).

---
## Detección

Cada frame del stream sigue este pipeline:

1. **YOLO** (`yolov8s-pose.pt`, conf ≥ 0.8, clase `person`) → lista de `Person`.
2. Cada `Person` lleva su bbox y 7 keypoints COCO seleccionados (nariz, ojos, orejas y hombros).
3. **Embedding ReID** opcional: si `models/osnet_x0_25_msmt17.pt` está disponible, cada bbox se recorta y se pasa por OSNet.
4. **Selección del objetivo activo** (`select_target_person`):
   - **Sin lock** → persona con bbox mayor.
   - **Con lock** → mínima distancia respecto al embedding fijado. Si la distancia supera `LOCK_MATCH_THRESHOLD = 0.30`, se considera pérdida del objetivo.
5. **Suavizado EMA** sobre la posición del punto (`alpha = 0.3`) y sobre el closeness (mismo alpha) para eliminar jitter.
6. **Histéresis de detección** (`TRACKING_MISS_HYSTERESIS_FRAMES = 45`): las caídas momentáneas de confianza no disparan la búsqueda.

---

## Multi-target lock (ReID)

Cuando hay **varias personas** en el frame, el dron debe seguir a una concreta y no saltar entre ellas. Solución: `torchreid`.

- Modelo: **OSNet x0_25** entrenado en **MSMT17** (`models/osnet_x0_25_msmt17.pt`).
- Cada detección se convierte en un vector 512-D unitario.
- `LockState.embedding` guarda la huella del objetivo escogido.
- En cada frame: `dist = 1 − cos(embedding_lock, embedding_i)`. La detección con menor distancia que pase el umbral se considera el objetivo.
- **Suavizado EMA del embedding** (`alpha = 0.05`): evita perder la fijación ante fluctuaciones puntuales.
- Si ninguna detección pasa el umbral → entra en `hover` (grace 4 s) → `searching`.

**Tres formas de adquirir la fijación:**

| Acción                     | Resultado                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Tecla `I` / botón «Lock»   | Sin objetivo seleccionado → fija el bbox más grande. Con objetivo fijado → rota entre las detecciones de izquierda a derecha. |
| Click sobre el vídeo (web) | Fija sobre la persona cuyo bbox contiene el punto cliqueado.                                                               |
| Tecla `C` / botón «Clear»  | Libera la fijación → vuelve a «bbox más grande».                                                                           |

---

## Sistema de objetivos

`Target` es una clase que describe **qué punto del cuerpo sigue el dron y cómo lo encuadra**. Vienen cuatro predefinidos:

| Target                              | Punto seguido             | Encuadre Y  | Señal de cercanía          | Setpoint |
| ----------------------------------- | ------------------------- | ----------- | -------------------------- | -------- |
| `nose`                              | Centro de la nariz        | 0.20 (alto) | bbox height / frame height | 0.7      |
| `eyes_midpoint`                     | Punto medio de ambos ojos | 0.20        | bbox height / frame height | 0.7      |
| `shoulders_midpoint` (por defecto)  | Centro de los hombros     | 0.25        | bbox width / frame width   | 0.45     |
| `bbox_center`                       | Centro del bbox completo  | 0.50        | bbox height / frame height | 0.7      |

Cada `Target` aporta:

- `is_visible(person)`: gate sobre los keypoints requeridos.
- `point(person)`: devuelve `(x, y)`; alimenta los PID de yaw, altitud y roll.
- `closeness(person)` y `closeness_setpoint`: usados por el **PID de pitch en modo bbox** cuando el ToF frontal no es válido.

El target es **intercambiable en tiempo de ejecución** desde la consola web (dropdown `target_name`). Al cambiar, se reinician las integrales y los EMAs para no propagar el estado de un target al siguiente.

---

## Control PID (4 ejes)

Cuatro bucles PID corren en paralelo a **20 Hz** (`RC_LOOP_INTERVAL_S = 0.05 s`):

| Eje | Variable controlada | Error | Ganancias `(Kp, Ki, Kd)` | Salida |
|---|---|---|---|---|
| **Altitud (baro)** | `height_cm` | `target_alt − height_cm` (cm) | `(1.4, 0.04, 0.08)` | `ud` cm/s (clamp ±60) |
| **Altitud (target)** | `target_y_px` | `target_y_smoothed − y_setpoint` (px) | `(−0.15, −0.004, −0.08)` | `ud` cm/s (clamp ±60) |
| **Yaw** | `target_x_px` | `target_x_smoothed − frame_center_x` (px) | `(0.2, 0.003, 0.1)` | `yaw` deg/s (clamp ±50) |
| **Pitch (ToF)** | distancia frontal | `intercept_dist − front_tof_cm` (cm) | `(−0.85, −0.02, −0.42)` | `fb` cm/s (clamp ±100) |
| **Pitch (bbox)** | closeness | `setpoint − closeness_smoothed` | `(200, 0, 20)` | `fb` cm/s (clamp ±100) |
| **Roll** | `target_x_px` (fuera de bbox) | `target_x − frame_center_x` (px) | `(0.2, 0.01, 0.15)` | `lr` cm/s (clamp ±60) |

**Convención de signo del error**: la altitud usa `target − medida` (eje Y de imagen invertido respecto al mundo arriba); el yaw usa `medida − target`. La discrepancia es **intencional** — no unificar sin verificar ambos ejes a la vez.

**Detalles importantes del bucle:**

- **Anti-windup** por saturación más reset explícito (`reset_integral()`) al cambiar de modo, de target o de fuente del PID de pitch.
- **Hand-off ToF ↔ bbox** en el eje de pitch: la histéresis `FRONT_TOF_INVALID_HYSTERESIS_FRAMES = 3` evita parpadeo en el borde de los 120 cm. Al cambiar de fuente, el `error_last` del PID entrante se inicializa con el error actual para que el término D no salte en la primera muestra (las unidades cambian de cm a ratio).
- **Roll con dead-zone de bbox**: si el centro horizontal de la imagen cae dentro del bbox suavizado del objetivo, `lr = 0`. Evita oscilaciones laterales innecesarias cuando el objetivo ya está «delante».

---
## Máquina de estados del vuelo

El bucle de vídeo selecciona un modo en cada iteración según `is_airborne`, `tracking_target` y el grace timer:
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
                  │                          │searching │ ── algoritmo de búsqueda
                  │                          └──────────┘
                  │
                  └── manual (M) ◀──▶ cualquier modo (override de RC)
```

- **grounded** — sin envío de RC; todas las integrales a cero.
- **intercepting** — todos los PID activos (altitud-target, yaw, pitch y roll).
- **hover** — `ud = 0` (el Tello se mantiene con barómetro + optical-flow). El yaw apunta hacia el último lado conocido si el objetivo se perdió por el borde del frame.
- **searching** — altitud-baro activa; el algoritmo de búsqueda controla yaw y fb.
- **manual** — `lr/fb/ud/yaw` provienen del teclado o del dpad web (con heartbeat de 0,6 s).

---

## Algoritmo de búsqueda

Cuando expira el grace de 4 s sin reencontrar el objetivo, el dron entra en `searching`.

```
        ┌─────────┐
   ──▶  │  spin   │  gira en yaw a SEARCH_SPIN_VELOCITY_DEG_S (50°/s)
        └────┬────┘  hacia el último lado conocido del objetivo.
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
   │    │ advance │  barrido el ToF lee ≤ 80 cm → muro en diagonal,
   │    │ sweeping│  aborta el advance y vuelve a spin.
   │    └────┬────┘
   │         │
   └─────────┘  Repite hasta que:
                · |error| de ToF < 15 cm (objetivo a distancia de interceptación)
                · timeout de 8 s
                · distancia integrada > 100 cm (cap por seguridad)
                → vuelve a spin
```

Si en cualquier momento se reencuentra el objetivo (`tracking_target == True`), el algoritmo de búsqueda se reinicia y se vuelve a `intercepting`.

---

## Capas de seguridad

Antes de cada envío de RC se aplican filtros **acumulables**:

1. **Geofence vertical** — bloquea `ud > 0` si `height_cm ≥ MAX_TRACKING_ALTITUDE_CM` (180 cm) y `ud < 0` si `≤ MIN_TRACKING_ALTITUDE_CM` (30 cm).
2. **Wall-stop** — si `front_tof_cm ≤ FRONT_TOF_WALL_STOP_CM` (60 cm) y `fb > 0`, fuerza `fb = 0` y reinicia las integrales del PID de pitch.
3. **Mitigación de pared diagonal** — al detectar la transición `front_tof_cm: válido → −1`, se congela `fb` durante `FRONT_TOF_DISCONTINUITY_FREEZE_S = 0.5 s`.
4. **Modo phantom** — desactiva por completo el envío de comandos al hardware (solo HUD).

---

## Consola Web (FastAPI)
Servidor FastAPI lanzado en un hilo daemon al arrancar `main.py`.

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

- **Sin autenticación ni HTTPS** — pensado para `localhost` o LAN sobre el AP del dron. Si se expone a WAN, hay que añadir túnel y token.
- **Heartbeat manual**: cada `/cmd/manual_set` desplaza `_web_manual_until` 0,6 s hacia el futuro. El frontend reenvía cada ~100 ms mientras se pulsa un botón del dpad. Si la pestaña muere, el heartbeat caduca y el dron se detiene.
- **Telemetría completa** vía WebSocket: `mode, is_airborne, is_manual, battery, front_tof, down_tof, height, pitch, roll, yaw, vx/vy/vz, ax/ay/az, temp, rc[4], locked_track_id, lock_distance, lock_target_idx, num_detections, reid_available, search_state, target_name, available_targets`.

---

## HUD OpenCV
La ventana local (`cv2.imshow("TelloInterceptor")`) muestra el frame con:

- Bboxes y keypoints de todas las detecciones.
- Resaltado diferenciado para el objetivo seleccionado y para el «objetivo fijado por ReID».
- Crosshair central y línea horizontal del setpoint Y del target activo.
- Tira de telemetría: batería, altitudes, ToFs, ángulos y RC actual.
- Indicadores: `PHANTOM` (si está activo), modo de control (`AUTO` / `MANUAL` con valores actuales) y aviso de pared.
- Etiqueta del estado de detección (target visible, modo actual y nombre del target).

---

## Modo Phantom

Editar `main.py` y poner `PHANTOM = True`:

```python
PHANTOM = True
```

- El dron **nunca despega** (se omite `takeoff()`).
- El RC loop no envía nada al hardware.
- Toda la lógica sigue corriendo: detección, PID, algoritmo de búsqueda, HUD y consola web.

Útil para ajustar ganancias y depurar sin batería ni espacio físico.

---
## Atajos de teclado

| Tecla | Acción |
|---|---|
| `SPACE` | Takeoff / land toggle |
| `M` | Entrar / salir de modo manual |
| `W / S / A / D` | Forward / back / left / right (manual) |
| `Q / E` | Yaw izquierda / derecha (manual) |
| `↑ / ↓` | Subir / bajar (manual) |
| `I` | Lock-cycle (bbox más grande → rota entre detecciones) |
| `C` | Liberar lock |
| `ESC` | Apagado limpio (guarda logs y aterriza) |

---

## Estructura del repositorio

```
proyecto-drones-tello/
├── main.py                    # entry point
├── interceptor/
│   ├── tello_interceptor.py   # orquestador + hilos
│   ├── constants.py           # única fuente de constantes
│   ├── perception.py          # YOLOv8-pose → Person
│   ├── target.py              # Target + LockState + selección
│   ├── reid.py                # OSNet embedder
│   ├── pid_controller.py      # PID genérico
│   └── hud_overlay.py         # dibujado sobre el frame
├── webapp/
│   ├── server.py              # FastAPI + WebSocket + MJPEG
│   └── static/index.html      # consola operador
├── models/
│   └── osnet_x0_25_msmt17.pt  # ReID (vendored)
├── yolov8s-pose.pt            # detector + pose (en la raíz)
├── requirements.txt
└── TelloInterceptor.pptx
```

---

## Limitaciones conocidas

- El dead-reckoning del Tello no es fiable; la geofence horizontal no se ha implementado.
- **ToF frontal** con alcance muy limitado.
- **YOLOv8s-pose @ 960 × 720** corre a ~30 FPS en GPU dedicada (no se ha probado en CPU).

---

## Licencia y créditos

Proyecto de Drones
Mayo de 2026
Pedro Fiol Benites (`Aszent47`).

Modelos preentrenados:
- YOLOv8s-pose — Ultralytics.
- OSNet x0_25 / MSMT17 — KaiyangZhou/deep-person-reid.

Librerías: `djitellopy`, `ultralytics`, `torchreid`, `fastapi`, `uvicorn`, `opencv-python` y PyTorch.

WebApp: backend implementado con Claude Code; frontend diseñado con claude.ai/design.
