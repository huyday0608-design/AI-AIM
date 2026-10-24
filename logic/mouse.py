import time
import math
import os
import supervision as sv

from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.logger import logger
from logic.platform import input_backend

if cfg.mouse_rzr:
    from logic.rzctl import RZCONTROL

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

class PIDController:
    """
    Precision Adaptive Tracking Controller with EMA Smoothing.
    Features:
    - Pure Proportional tracking with critically damped Exponential Moving Average.
    - Zero integral windup (prevents swinging / limit cycle oscillation).
    - Zero derivative noise amplification (prevents visual chatter).
    - Sub-pixel remainder accumulator to prevent quantization deadzones.
    """
    def __init__(self, kp: float = 0.80, ki: float = 0.0, kd: float = 0.0, max_integral: float = 30.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_integral = float(max_integral)
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.remainder = 0.0
        self.smooth_output = 0.0

    def update(self, error: float, current_time: float = None) -> float:
        if current_time is None:
            current_time = time.time()

        if self.prev_time is None or (current_time - self.prev_time) > 0.2:
            dt = 0.016
            derivative = 0.0
            self.smooth_output = 0.0
        else:
            dt = max(0.001, min(0.080, current_time - self.prev_time))
            derivative = (error - self.prev_error) / dt

        # Zero-crossing anti-windup: reset integral when error crosses zero to eliminate overshoot
        if (error > 0 and self.prev_error < 0) or (error < 0 and self.prev_error > 0):
            self.integral = 0.0

        if self.ki > 0:
            self.integral = max(-self.max_integral, min(self.max_integral, self.integral + error * dt))
        else:
            self.integral = 0.0

        raw_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Critically damped smoothing: loại bỏ triệt để rung lắc do jitter AI, bám dính êm ru vào zombie di chuyển
        alpha = 0.45
        self.smooth_output = alpha * raw_output + (1.0 - alpha) * self.smooth_output

        self.prev_error = error
        self.prev_time = current_time

        total = self.smooth_output + self.remainder
        step = int(total)
        self.remainder = total - step

        return float(step)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.remainder = 0.0
        self.smooth_output = 0.0

    def update_constants(self, kp: float, ki: float, kd: float, max_integral: float = 50.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_integral = float(max_integral)


class MouseThread:
    def __init__(self):
        self.initialize_parameters()
        self.setup_hardware()

    def initialize_parameters(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.prev_distance = None
        self.speed_correction_factor = 0.1
        self.bScope = False
        self.arch = self.get_arch()
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        self._warned_input_unavailable = False
        self.spray_time = 0.0
        self.last_shot_time = None

        if not hasattr(self, 'pid_x'):
            self.pid_x = PIDController(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
            self.pid_y = PIDController(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
        else:
            self.pid_x.update_constants(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
            self.pid_y.update_constants(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
            self.reset_pid()

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        ai_device = str(cfg.AI_device).lower()
        if 'cpu' in ai_device:
            return 'cpu'
        if ai_device == 'cuda':
            return 'cuda'
        if ai_device.startswith('cuda:'):
            return cfg.AI_device
        return f'cuda:{cfg.AI_device}'

    def setup_hardware(self):
        if cfg.mouse_ghub and not hasattr(self, 'ghub'):
            from logic.ghub import gHub
            self.ghub = gHub

        if cfg.mouse_rzr and not hasattr(self, 'rzr'):
            from logic.rzctl import RZCONTROL
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rzctl.dll")
            self.rzr = RZCONTROL(dll_path)
            if not self.rzr.init():
                logger.error("Failed to initialize rzctl")

    def process_data(self, data):
        if isinstance(data, sv.Detections):
            xyxy = data.xyxy[0]
            target_x = (xyxy[0] + xyxy[2]) / 2
            target_y = (xyxy[1] + xyxy[3]) / 2
            target_w = xyxy[2] - xyxy[0]
            target_h = xyxy[3] - xyxy[1]
            target_cls = data.class_id[0] if data.class_id is not None and data.class_id.size > 0 else None
            vel_x = 0.0
            vel_y = 0.0
        elif len(data) >= 7:
            target_x, target_y, target_w, target_h, target_cls, vel_x, vel_y = data[:7]
        else:
            target_x, target_y, target_w, target_h, target_cls = data
            vel_x = 0.0
            vel_y = 0.0

        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        shooting_state = self.get_shooting_key_state() or is_firing
        if not (shooting_state or cfg.mouse_auto_aim):
            self.reset_pid()
        self.visualize_target(target_x, target_y, target_cls)
        self.bScope = self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bScope_multiplier) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        # Stable Target Anchoring: Trong hệ quy chiếu camera di động, điểm ngắm bám trực tiếp vào
        # tọa độ mục tiêu thực tế (target_x, target_y).
        # Loại bỏ hoàn toàn việc cộng dồn (vel_x * lead_time) nhằm triệt tiêu phản hồi âm 180 độ
        # gây ra hiện tượng rung lắc chuột loạn xạ và vẩy sang tường.
        effective_x = target_x
        effective_y = target_y

        if not self.disable_prediction:
            current_time = time.time()
            if not isinstance(data, sv.Detections):
                effective_x, effective_y = self.predict_target_position(effective_x, effective_y, current_time)
            self.visualize_prediction(effective_x, effective_y, target_cls)

        move_x, move_y = self.calc_movement(effective_x, effective_y, target_cls, shooting_state, target_w, target_h)

        self.visualize_history(target_x, target_y)
        shooting.submit(self.bScope, shooting_state)
        self.move_mouse(move_x, move_y, shooting_state)

    def predict_target_position(self, target_x, target_y, current_time):
        # First target
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_x = target_x
            self.prev_y = target_y
            self.prev_velocity_x = 0
            self.prev_velocity_y = 0
            return target_x, target_y

        # Next target?
        max_jump = max(self.screen_width, self.screen_height) * 0.3 # 30%
        if abs(target_x - self.prev_x) > max_jump or abs(target_y - self.prev_y) > max_jump:
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_velocity_x = 0
            self.prev_velocity_y = 0
            self.prev_time = current_time
            self.reset_pid()
            return target_x, target_y

        delta_time = current_time - self.prev_time

        if delta_time <= 0:
            delta_time = 1e-3

        velocity_x = (target_x - self.prev_x) / delta_time
        velocity_y = (target_y - self.prev_y) / delta_time
        acceleration_x = (velocity_x - self.prev_velocity_x) / delta_time
        acceleration_y = (velocity_y - self.prev_velocity_y) / delta_time

        prediction_interval = delta_time * self.prediction_interval
        current_distance = math.sqrt((target_x - self.prev_x)**2 + (target_y - self.prev_y)**2)
        proximity_factor = max(0.1, min(1, 1 / (current_distance + 1)))

        speed_correction = 1 + (abs(current_distance - (self.prev_distance or 0)) / self.max_distance) * self.speed_correction_factor if self.prev_distance is not None else 1.0

        # Safety displacement clamp: dự đoán không được vượt quá 12px để tránh hiện tượng vung chuột nhảy sang con thứ 2
        pred_offset_x = max(-12.0, min(12.0, velocity_x * prediction_interval * proximity_factor * speed_correction + 0.5 * acceleration_x * (prediction_interval ** 2) * proximity_factor * speed_correction))
        pred_offset_y = max(-12.0, min(12.0, velocity_y * prediction_interval * proximity_factor * speed_correction + 0.5 * acceleration_y * (prediction_interval ** 2) * proximity_factor * speed_correction))

        predicted_x = target_x + pred_offset_x
        predicted_y = target_y + pred_offset_y

        self.prev_x, self.prev_y = target_x, target_y
        self.prev_velocity_x, self.prev_velocity_y = velocity_x, velocity_y
        self.prev_time = current_time
        self.prev_distance = current_distance

        return predicted_x, predicted_y

    def calculate_speed_multiplier(self, target_x, target_y, distance):
        if any(map(math.isnan, (target_x, target_y))) or self.section_size_x == 0:
            return self.min_speed_multiplier

        normalized_distance = min(distance / self.max_distance, 1)
        base_speed = self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * (1 - normalized_distance)

        if self.section_size_x == 0:
            return self.min_speed_multiplier

        target_x_section = int((target_x - self.center_x + self.screen_width / 2) / self.section_size_x)
        target_y_section = int((target_y - self.center_y + self.screen_height / 2) / self.section_size_y)

        distance_from_center = max(abs(50 - target_x_section), abs(50 - target_y_section))

        if distance_from_center == 0:
            return 1
        elif 5 <= distance_from_center <= 10:
            return self.max_speed_multiplier
        else:
            speed_reduction = max(0, min(distance_from_center - 10, 45)) / 100.0
            speed_multiplier = base_speed * (1 - speed_reduction)

        if self.prev_distance is not None:
            speed_adjustment = 1 + (abs(distance - self.prev_distance) / self.max_distance) * self.speed_correction_factor
            return speed_multiplier * speed_adjustment

        return speed_multiplier

    def _update_spray_state(self) -> bool:
        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        now = time.time()
        if is_firing:
            if self.last_shot_time is None or (now - self.last_shot_time) > 0.25:
                self.spray_time = 0.0
            else:
                self.spray_time += (now - self.last_shot_time)
            self.last_shot_time = now
        else:
            self.last_shot_time = None
            self.spray_time = 0.0
        return is_firing

    def calc_movement(self, target_x, target_y, target_cls, shooting_state=False, target_w=0.0, target_h=0.0):
        # Coordinate sanity & boundary validation
        if not (isinstance(target_x, (int, float)) and isinstance(target_y, (int, float))):
            return 0.0, 0.0
        if not (math.isfinite(target_x) and math.isfinite(target_y)):
            return 0.0, 0.0
        if target_x < 0 or target_x > self.screen_width or target_y < 0 or target_y > self.screen_height:
            return 0.0, 0.0

        is_firing = self._update_spray_state()
        is_sniper = getattr(cfg, 'weapon_is_sniper', False)

        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        distance = math.hypot(offset_x, offset_y)

        # Smooth Asymptotic Approach & Anti-Deadzone:
        # 1. Khi ở cự ly vi mô trong deadzone: Dừng hoàn toàn để tâm khóa bất động trên đầu mục tiêu
        if distance <= cfg.mouse_deadzone:
            return 0.0, 0.0

        # 2. Khi ở gần tâm (< 16px): Giảm tốc mượt mà nhưng duy trì lực kéo chắc chắn (>= 0.60) để tâm ăn thẳng vào đầu
        if distance < 16.0:
            norm = (distance - cfg.mouse_deadzone) / max(1.0, 16.0 - cfg.mouse_deadzone)
            smooth_factor = 0.60 + 0.40 * norm
        else:
            smooth_factor = 1.0

        degrees_per_pixel_x = self.fov_x / self.screen_width
        degrees_per_pixel_y = self.fov_y / self.screen_height
        sensitivity = max(self.mouse_sensitivity, 1e-6)

        scale_x = (degrees_per_pixel_x / 360.0) * (self.dpi / sensitivity)
        scale_y = (degrees_per_pixel_y / 360.0) * (self.dpi / sensitivity)

        raw_move_x = offset_x * scale_x
        raw_move_y = offset_y * scale_y

        now = time.time()
        move_x = self.pid_x.update(raw_move_x * smooth_factor, current_time=now)
        move_y = self.pid_y.update(raw_move_y * smooth_factor, current_time=now)

        # Trigger-Coupled Drag: Trợ lực ghìm tâm lũy tiến khi xả súng liên thanh (không kích hoạt cho súng ngắm)
        if not is_sniper and cfg.recoil_compensation and is_firing:
            if self.spray_time < 0.16:
                pull_stage = (self.spray_time / 0.16) * 0.35
            elif self.spray_time < 0.55:
                p = (self.spray_time - 0.16) / (0.55 - 0.16)
                pull_stage = 0.35 + 0.50 * (3.0 * p**2 - 2.0 * p**3)
            else:
                pull_stage = 0.85 + 0.15 * min(1.0, (self.spray_time - 0.55) / 0.45)

            direct_pull = pull_stage * cfg.recoil_pitch * 0.90
            direct_pull = min(direct_pull, cfg.recoil_max_pitch)
            move_y += direct_pull * (scale_y / 1.5)

        return move_x, move_y

    def get_recoil_offset(self, shooting_state=False):
        # Used for standalone recoil pull-down when spraying freely into open space
        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        if is_firing:
            recoil_y = min(
                cfg.recoil_pitch * (1.0 + self.spray_time * cfg.recoil_spray_scale),
                cfg.recoil_max_pitch
            )
            return recoil_y
        return 0.0

    def apply_standalone_recoil(self):
        """Applies recoil pull-down when player sprays freely without locking on a target."""
        if not cfg.recoil_compensation:
            return
        recoil_y = self.get_recoil_offset()
        if recoil_y > 0:
            self.move_mouse(0, int(recoil_y), True)

    def reset_pid(self):
        if hasattr(self, 'pid_x'):
            self.pid_x.reset()
        if hasattr(self, 'pid_y'):
            self.pid_y.reset()
        if hasattr(self, 'last_move_x'):
            self.last_move_x = 0.0
            self.last_move_y = 0.0
        if hasattr(self, 'remainder_x'):
            self.remainder_x = 0.0
            self.remainder_y = 0.0
        self.spray_time = 0.0
        self.last_shot_time = None

    def move_mouse(self, x, y, shooting_state):
        # Emergency pause guard: when paused, do not send any movement!
        try:
            from logic.hotkeys_watcher import hotkeys_watcher
            if hotkeys_watcher.app_pause != 0:
                self.reset_pid()
                return
        except Exception:
            pass

        # Dry-run simulation mode: do not send mouse events, safe for offline benchmarking
        if getattr(cfg, 'mouse_dry_run', False):
            return

        move_x = int(x)
        move_y = int(y)

        # Max step safety clamp: giới hạn tối đa 25px/khung hình để loại bỏ hoàn toàn giật văng chuột
        max_step = min(cfg.mouse_max_step, 25) if cfg.mouse_max_step > 0 else 25
        move_x = max(-max_step, min(max_step, move_x))
        move_y = max(-max_step, min(max_step, move_y))

        # Slew-rate limiter / max acceleration safety clamp:
        # Prevents violent mouse twitching between consecutive frames
        if hasattr(self, 'last_sent_move_x') and hasattr(self, 'last_sent_move_y'):
            max_accel = 16
            move_x = max(self.last_sent_move_x - max_accel, min(self.last_sent_move_x + max_accel, move_x))
            move_y = max(self.last_sent_move_y - max_accel, min(self.last_sent_move_y + max_accel, move_y))
        self.last_sent_move_x = move_x
        self.last_sent_move_y = move_y

        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        active_aiming = shooting_state or cfg.mouse_auto_aim

        # Horizontal tracking requires active aiming intent.
        # Merely firing left click does NOT jerk crosshair horizontally!
        if not active_aiming:
            move_x = 0

        if move_x == 0 and move_y == 0:
            return

        if active_aiming or (cfg.recoil_compensation and is_firing):
            if not cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                if not input_backend.move_mouse(move_x, move_y):
                    if not self._warned_input_unavailable:
                        logger.warning("[Mouse] Native mouse movement backend is unavailable on this platform.")
                        self._warned_input_unavailable = True
            elif cfg.mouse_ghub:
                self.ghub.mouse_xy(move_x, move_y)
            elif cfg.arduino_move:
                from logic.arduino import arduino
                arduino.move(move_x, move_y)
            elif cfg.mouse_rzr:
                self.rzr.mouse_move(move_x, move_y, True)

    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            if input_backend.is_pressed(key_name.strip()):
                return True
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        if getattr(cfg, 'weapon_is_sniper', False):
            # Sniper Triggerbot: Thu hẹp vùng ngắm siêu chuẩn vào đúng hộp sọ (Head Hitbox)
            # Chỉ kích hoạt bắn khi tâm ngắm đã nằm chính xác trong bán kính đầu
            head_r = max(6.0, target_h * 0.18 * reduction_factor)
            bScope = math.hypot(self.center_x - target_x, self.center_y - target_y) <= head_r
            if cfg.show_window and cfg.show_bScope_box:
                visuals.draw_bScope(target_x - head_r, target_x + head_r, target_y - head_r, target_y + head_r, bScope)
            return bScope

        reduced_w, reduced_h = target_w * reduction_factor / 2, target_h * reduction_factor / 2
        x1, x2, y1, y2 = target_x - reduced_w, target_x + reduced_w, target_y - reduced_h, target_y + reduced_h
        bScope = self.center_x > x1 and self.center_x < x2 and self.center_y > y1 and self.center_y < y2

        if cfg.show_window and cfg.show_bScope_box:
            visuals.draw_bScope(x1, x2, y1, y2, bScope)

        return bScope

    def update_settings(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        self.arch = self.get_arch()
        self.prev_time = None
        self.prev_distance = None
        if hasattr(self, 'pid_x'):
            self.pid_x.update_constants(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
            self.pid_y.update_constants(cfg.mouse_pid_kp, cfg.mouse_pid_ki, cfg.mouse_pid_kd, cfg.mouse_pid_max_integral)
            self.reset_pid()
        self.setup_hardware()

    def visualize_target(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.overlay_show_target_line):
            visuals.draw_target_line(target_x, target_y, target_cls)

    def visualize_prediction(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.overlay_show_target_prediction_line):
            visuals.draw_predicted_position(target_x, target_y, target_cls)

    def visualize_history(self, target_x, target_y):
        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(target_x, target_y)

mouse = MouseThread()
