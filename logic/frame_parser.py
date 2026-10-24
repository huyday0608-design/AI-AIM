import time
import math
import torch
import supervision as sv
import numpy as np

from logic.hotkeys_watcher import hotkeys_watcher
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse
from logic.shooting import shooting
from logic.platform import input_backend
from logic.model_classes import HEAD_CLASS_ID, is_head_class


class TrackingState:
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    TEMPORARY_LOST = "TEMPORARY_LOST"
    LOST = "LOST"


class Target:
    def __init__(self, x, y, w, h, cls, track_id=None, aim_x=None, aim_y=None, vel_x=0.0, vel_y=0.0):
        self.cls = int(cls)
        self.w = float(w)
        self.h = float(h)
        self.track_id = track_id
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        if aim_x is not None and aim_y is not None:
            self.x = float(aim_x)
            self.y = float(aim_y)
        else:
            self.x = float(x)
            is_sniper = getattr(cfg, 'weapon_is_sniper', False)
            body_offset = 0.46 if is_sniper else 0.42
            self.y = float(y - 0.15 * self.h) if is_head_class(self.cls) else float(y - body_offset * self.h)


class LockedZombieTrack:
    """
    Robust DIoU + Velocity Kalman-style Tracklet for moving targets.
    - Tracks position, bounding box, relative head aimpoint, and velocity vector.
    - DIoU matching guarantees moving zombies are continuously tracked without jumping to nearby zombies.
    - Preserves relative head aimpoint even if head detection flickers off.
    - Coasts smoothly along velocity trajectory during muzzle flash / smoke.
    """
    def __init__(self, box_x, box_y, box_w, box_h, aim_x, aim_y, cls_id, track_id=None):
        self.box_x = float(box_x)
        self.box_y = float(box_y)
        self.box_w = float(box_w)
        self.box_h = float(box_h)
        self.aim_x = float(aim_x)
        self.aim_y = float(aim_y)
        self.cls_id = int(cls_id)
        self.track_id = track_id
        self.rel_aim_x = (self.aim_x - self.box_x) / max(self.box_w, 1.0)
        self.rel_aim_y = (self.aim_y - self.box_y) / max(self.box_h, 1.0)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.has_velocity = False
        self.last_time = time.time()

    def update(self, box_x, box_y, box_w, box_h, aim_x, aim_y, track_id=None, now=None, cls_id=None):
        if now is None:
            now = time.time()
        dt = max(0.001, min(0.15, now - self.last_time))

        inst_vx = (box_x - self.box_x) / dt
        inst_vy = (box_y - self.box_y) / dt
        inst_vx = max(-350.0, min(350.0, inst_vx))
        inst_vy = max(-350.0, min(350.0, inst_vy))

        if not self.has_velocity:
            self.vel_x = inst_vx
            self.vel_y = inst_vy
            self.has_velocity = True
        else:
            alpha = 0.50
            self.vel_x = alpha * inst_vx + (1.0 - alpha) * self.vel_x
            self.vel_y = alpha * inst_vy + (1.0 - alpha) * self.vel_y

        self.box_x = float(box_x)
        self.box_y = float(box_y)
        self.box_w = 0.7 * self.box_w + 0.3 * float(box_w)
        self.box_h = 0.7 * self.box_h + 0.3 * float(box_h)

        # Target aimpoint positioning:
        is_sniper = getattr(cfg, 'weapon_is_sniper', False)
        if is_sniper:
            # Sniper Mode: Khóa tuyệt đối vào điểm đầu (100% Headshot), snap tức thì không delay
            if aim_x is not None and aim_y is not None:
                self.aim_x = float(aim_x)
                self.aim_y = float(aim_y)
            else:
                self.aim_x = self.box_x + self.rel_aim_x * self.box_w
                self.aim_y = self.box_y + self.rel_aim_y * self.box_h
        else:
            # Smooth aimpoint with clamp to eliminate vertical head/body alternation jitter:
            if aim_x is not None and aim_y is not None:
                clamped_dx = max(-15.0, min(15.0, float(aim_x) - self.aim_x))
                clamped_dy = max(-15.0, min(15.0, float(aim_y) - self.aim_y))
                self.aim_x += 0.50 * clamped_dx
                self.aim_y += 0.50 * clamped_dy
            else:
                self.aim_x = self.box_x + self.rel_aim_x * self.box_w
                self.aim_y = self.box_y + self.rel_aim_y * self.box_h

        self.rel_aim_x = (self.aim_x - self.box_x) / max(self.box_w, 1.0)
        self.rel_aim_y = (self.aim_y - self.box_y) / max(self.box_h, 1.0)

        if track_id is not None:
            self.track_id = track_id
        if cls_id is not None:
            self.cls_id = int(cls_id)
        self.last_time = now

    def predict(self, now=None):
        # Stable holdover: giữ nguyên vị trí hộp mục tiêu ổn định, không bay lệch vào tường
        return self.box_x, self.box_y, self.box_w, self.box_h, self.aim_x, self.aim_y

    def compute_diou(self, box_x, box_y, box_w, box_h, now=None):
        pred_bx, pred_by, pred_bw, pred_bh, _, _ = self.predict(now)
        x1_a, y1_a = pred_bx - pred_bw / 2.0, pred_by - pred_bh / 2.0
        x2_a, y2_a = pred_bx + pred_bw / 2.0, pred_by + pred_bh / 2.0
        x1_b, y1_b = box_x - box_w / 2.0, box_y - box_h / 2.0
        x2_b, y2_b = box_x + box_w / 2.0, box_y + box_h / 2.0

        ix1 = max(x1_a, x1_b)
        iy1 = max(y1_a, y1_b)
        ix2 = min(x2_a, x2_b)
        iy2 = min(y2_a, y2_b)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih
        union_area = (pred_bw * pred_bh) + (box_w * box_h) - inter_area
        iou = inter_area / max(union_area, 1e-6)

        c_x1 = min(x1_a, x1_b)
        c_y1 = min(y1_a, y1_b)
        c_x2 = max(x2_a, x2_b)
        c_y2 = max(y2_a, y2_b)
        c_diag_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2
        dist_sq = (pred_bx - box_x)**2 + (pred_by - box_y)**2
        diou = iou - (dist_sq / max(c_diag_sq, 1e-6))
        return diou


class FrameParser:
    def __init__(self):
        self.arch = self.get_arch()
        self._center = None
        self._center_cache_key = None
        # Target Lock / Persistence state
        self.locked_track = None
        self.locked_pos = None
        self.locked_track_id = None
        self.locked_time = 0.0
        self.last_aim_time = 0.0
        self.locked_w = 0.0
        self.locked_h = 0.0
        self.locked_cls = None
        self.lock_timeout = 0.45  # 450ms grace period trước khi xác nhận mục tiêu chết hẳn
        self.flash_holdover = 0.30  # 300ms bridge for muzzle flash / shake / frame loss
        self.tracking_state = TrackingState.SEARCHING

    def parse(self, result):
        if result is None:
            self._handle_no_detections()
            return

        if isinstance(result, sv.Detections):
            self._process_sv_detections(result)
        else:
            self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        if len(detections) > 0:
            if cfg.show_window or cfg.show_overlay:
                visuals.draw_helpers(detections)
            target = self.sort_targets(detections)
            self._handle_target(target)
        else:
            self._handle_no_detections()

    def _process_yolo_detections(self, results):
        frames = (results,) if hasattr(results, "boxes") else results
        processed = False

        for frame in frames:
            processed = True
            boxes = getattr(frame, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                target = self.sort_targets(frame)
                self._handle_target(target)
                self._visualize_frame(frame)
            else:
                self._handle_no_detections()

        if not processed:
            self._handle_no_detections()

    def _handle_target(self, target):
        if target:
            if hotkeys_watcher.clss is None:
                hotkeys_watcher.active_classes()

            if target.cls in hotkeys_watcher.clss:
                mouse.process_data((target.x, target.y, target.w, target.h, target.cls, target.vel_x, target.vel_y))
                if self.locked_track is not None:
                    visuals.draw_locked_target(
                        self.locked_track.box_x, self.locked_track.box_y,
                        self.locked_track.box_w, self.locked_track.box_h,
                        self.locked_track.aim_x, self.locked_track.aim_y,
                        status=self.tracking_state
                    )

    def _visualize_frame(self, frame):
        if cfg.show_window or cfg.show_overlay:
            if cfg.show_boxes or cfg.overlay_show_boxes:
                visuals.draw_helpers(frame.boxes)

            if cfg.show_window and cfg.show_detection_speed:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])

    def _handle_no_detections(self):
        now = time.time()
        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")

        # Khi không có mục tiêu: KHÔNG tự động di chuột đẩy vào tường!
        # Reset PID controller để chuột đứng yên ổn định.
        mouse.reset_pid()

        # Reset lock if no targets seen within the active timeout period
        active_timeout = 0.40 if is_firing else self.lock_timeout
        if self.locked_track is not None:
            if (now - self.locked_track.last_time <= active_timeout):
                # Target is temporarily occluded (muzzle flash / smoke):
                # Maintain locked target visual indicator so UI doesn't blink!
                self.tracking_state = TrackingState.TEMPORARY_LOST
                visuals.draw_locked_target(
                    self.locked_track.box_x, self.locked_track.box_y,
                    self.locked_track.box_w, self.locked_track.box_h,
                    self.locked_track.aim_x, self.locked_track.aim_y,
                    status=TrackingState.TEMPORARY_LOST
                )
            else:
                self.tracking_state = TrackingState.LOST
                self.locked_track = None
                self.locked_pos = None
                self.locked_track_id = None
                self.locked_cls = None
                visuals.clear_locked_target()
                if cfg.show_window or cfg.show_overlay:
                    visuals.clear()
        else:
            self.tracking_state = TrackingState.SEARCHING
            if cfg.show_window or cfg.show_overlay:
                visuals.clear()

        if cfg.recoil_compensation and is_firing:
            mouse.apply_standalone_recoil()
        if cfg.auto_shoot or cfg.triggerbot:
            shooting.shoot(False, False)

    def sort_targets(self, frame):
        tracker_ids = None
        if hasattr(frame, "orig_shape"):
            self._current_frame_shape = frame.orig_shape
        elif hasattr(frame, "boxes") and hasattr(frame.boxes, "orig_shape"):
            self._current_frame_shape = frame.boxes.orig_shape
        else:
            self._current_frame_shape = (cfg.detection_window_height, cfg.detection_window_width)

        if isinstance(frame, sv.Detections):
            boxes_array, classes_tensor = self._convert_sv_to_tensor(frame)
            tracker_ids = getattr(frame, "tracker_id", None)
        else:
            if frame.boxes is None or len(frame.boxes) == 0:
                return None
            boxes_array = self._to_tensor(frame.boxes.xywh, dtype=torch.float32)
            classes_tensor = self._to_tensor(frame.boxes.cls, dtype=torch.long)
            tracker_ids = getattr(frame.boxes, "id", None) if hasattr(frame, "boxes") else None

        if not classes_tensor.numel():
            return None

        return self._find_nearest_target(boxes_array, classes_tensor, tracker_ids)

    def _convert_sv_to_tensor(self, frame):
        xyxy = np.asarray(frame.xyxy, dtype=np.float32)
        xywh = np.empty((xyxy.shape[0], 4), dtype=np.float32)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        boxes_tensor = torch.as_tensor(xywh, dtype=torch.float32, device=self.arch)
        class_ids = frame.class_id if frame.class_id is not None else np.zeros(xyxy.shape[0], dtype=np.int64)
        classes_tensor = torch.as_tensor(class_ids, dtype=torch.long, device=self.arch)
        return boxes_tensor, classes_tensor

    def _to_tensor(self, value, dtype):
        if hasattr(value, "to"):
            return value.to(device=self.arch, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.arch)

    def _find_nearest_target(self, boxes_array, classes_tensor, tracker_ids=None):
        candidate_idxs = self._get_active_candidate_idxs(classes_tensor)

        now = time.time()
        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        is_aiming = mouse.get_shooting_key_state() or cfg.mouse_auto_aim or is_firing
        if is_aiming:
            self.last_aim_time = now
        else:
            # Tap-fire grace period: Nhấp nhả chuột không làm mất khóa mục tiêu
            if (now - self.last_aim_time) > 0.40 and not (cfg.show_window or cfg.show_overlay):
                self.locked_track = None
                self.locked_pos = None
                self.locked_track_id = None

        if candidate_idxs.numel() == 0:
            if self.locked_track is not None and (now - self.locked_track.last_time) > self.lock_timeout:
                self.locked_track = None
                self.locked_pos = None
                self.locked_track_id = None
            return None

        # Extract boxes and classes for candidate detections
        cand_boxes = boxes_array[candidate_idxs, :4].cpu().numpy()
        cand_classes = classes_tensor[candidate_idxs].cpu().numpy()

        bodies = []
        heads = []

        for idx, (b, cls_id) in enumerate(zip(cand_boxes, cand_classes)):
            tid = None
            if tracker_ids is not None:
                try:
                    t = tracker_ids[candidate_idxs[idx].item()]
                    tid = t.item() if hasattr(t, "item") else t
                    if tid == -1:
                        tid = None
                except (IndexError, TypeError):
                    pass

            if cls_id == HEAD_CLASS_ID:
                heads.append({'box': b, 'cls': int(cls_id), 'tid': tid})
            else:
                bodies.append({'box': b, 'cls': int(cls_id), 'tid': tid})

        matched_head_indices = set()
        targets = []

        # 1. Match heads to bodies to form complete physical targets with headshot aimpoints
        if hasattr(self, '_current_frame_shape') and self._current_frame_shape:
            win_h, win_w = float(self._current_frame_shape[0]), float(self._current_frame_shape[1])
        else:
            win_w = float(cfg.detection_window_width)
            win_h = float(cfg.detection_window_height)

        for b in bodies:
            bx, by, bw, bh = b['box']
            # Reject full-screen wall / texture hallucination glitches (> 85% screen size)
            if bw > 0.85 * win_w and bh > 0.85 * win_h:
                continue

            matched_head = None
            is_sniper = getattr(cfg, 'weapon_is_sniper', False)
            if not cfg.disable_headshot:
                head_x_dev = 0.75 * bw if is_sniper else 0.60 * bw
                head_min_y = (by - 0.75 * bh) if is_sniper else (by - 0.60 * bh)
                head_max_y = (by + 0.20 * bh) if is_sniper else (by + 0.15 * bh)
                for h_idx, h in enumerate(heads):
                    hx, hy, hw, hh = h['box']
                    # Head lies in the upper section of the body bounding box
                    if abs(hx - bx) <= head_x_dev and head_min_y <= hy <= head_max_y:
                        matched_head = h
                        matched_head_indices.add(h_idx)
                        break

            if matched_head is not None:
                hx, hy, hw, hh = matched_head['box']
                # Pinpoint Headshot: Aim directly at the forehead / eye line (15% above center)
                head_aim_y = float(hy - 0.15 * hh)
                targets.append({
                    'aim_x': float(hx),
                    'aim_y': head_aim_y,
                    'box_x': float(bx),
                    'box_y': float(by),
                    'box_w': float(bw),
                    'box_h': float(bh),
                    'cls': HEAD_CLASS_ID,
                    'eff_h': float(bh),
                    'track_id': b['tid'] or matched_head['tid']
                })
            else:
                # Fallback: khi chỉ phát hiện thân (chưa phát hiện đầu riêng), đưa tâm thẳng lên đỉnh đầu
                body_offset = 0.47 if is_sniper else (0.42 if not cfg.disable_headshot else cfg.body_y_offset)
                aim_y = float(by - body_offset * bh)
                targets.append({
                    'aim_x': float(bx),
                    'aim_y': aim_y,
                    'box_x': float(bx),
                    'box_y': float(by),
                    'box_w': float(bw),
                    'box_h': float(bh),
                    'cls': HEAD_CLASS_ID if is_sniper else b['cls'],
                    'eff_h': float(bh),
                    'track_id': b['tid']
                })

        # 2. Add legitimate unmatched heads (e.g. enemy behind cover with only head visible)
        # Lọc triệt để đồ vật môi trường (ống nước, đèn trần, van xả, mô hình súng/tay của người chơi):
        if not cfg.disable_headshot:
            is_sniper = getattr(cfg, 'weapon_is_sniper', False)
            for h_idx, h in enumerate(heads):
                if h_idx not in matched_head_indices:
                    hx, hy, hw, hh = h['box']

                    # Bộ lọc 1: Hình dạng đầu người. Khi zoom kính ngắm Sniper, vạch ngang của tâm ngắm cắt qua đầu tạo thành hộp dẹt
                    max_ratio = 2.2 if is_sniper else 1.40
                    if hw > max_ratio * hh:
                        continue

                    # Bộ lọc 2: Kích thước hợp lý. Khi zoom ống ngắm Sniper, đầu có thể chiếm tới 65% màn hình FOV.
                    max_w = 0.65 * win_w if is_sniper else 0.40 * win_w
                    max_h = 0.70 * win_h if is_sniper else 0.45 * win_h
                    if hw > max_w or hh > max_h:
                        continue

                    # Bộ lọc 3: Vùng vũ khí của bản thân (Player Weapon Deadzone).
                    # Trong chế độ súng ngắm (zoom ADS), mô hình súng không che chắn góc màn hình.
                    if not is_sniper and (hy > 0.68 * win_h and hx > 0.32 * win_w):
                        continue

                    head_aim_y = float(hy - 0.15 * hh)
                    targets.append({
                        'aim_x': float(hx),
                        'aim_y': head_aim_y,
                        'box_x': float(hx),
                        'box_y': float(hy),
                        'box_w': float(hw),
                        'box_h': float(hh),
                        'cls': HEAD_CLASS_ID,
                        'eff_h': float(hh * 4.0),
                        'track_id': h['tid']
                    })

        if not targets:
            return None

        # 3. Dedicated Target Lock Commitment & Continuous Motion Tracking:
        selected = None
        aim_active = is_aiming or (now - self.last_aim_time <= 0.40) or (cfg.show_window or cfg.show_overlay)
        has_active_lock = (
            cfg.mouse_lock_target
            and aim_active
            and self.locked_track is not None
            and (now - self.locked_track.last_time <= self.lock_timeout)
        )

        if has_active_lock:
            # Priority A: Match by persistent ByteTrack tracker_id if available
            if self.locked_track.track_id is not None:
                for t in targets:
                    if t['track_id'] is not None and t['track_id'] == self.locked_track.track_id:
                        selected = t
                        break

            # Priority B: Spatial proximity matching to existing locked target
            if selected is None:
                candidates = []
                for t in targets:
                    d = math.hypot(t['box_x'] - self.locked_track.box_x, t['box_y'] - self.locked_track.box_y)
                    # Hạn chế độ dời tối đa giữa 2 khung hình để không bao giờ nhảy sang dummy kế bên:
                    max_allowed_dist = max(30.0, self.locked_track.box_w * 0.90)
                    if d <= max_allowed_dist:
                        candidates.append((d, t))
                if candidates:
                    candidates.sort(key=lambda item: item[0])
                    selected = candidates[0][1]

            # Khi mục tiêu bị che khuất tạm thời (chớp lửa đạn/khói bụi < lock_timeout):
            # Giữ nguyên khóa mục tiêu hiện tại, KHÔNG nhảy lung tung sang dummy khác!
            if selected is None:
                self.tracking_state = TrackingState.TEMPORARY_LOST
                mouse.reset_pid()
                visuals.draw_locked_target(
                    self.locked_track.box_x, self.locked_track.box_y,
                    self.locked_track.box_w, self.locked_track.box_h,
                    self.locked_track.aim_x, self.locked_track.aim_y,
                    status=TrackingState.TEMPORARY_LOST
                )
                return None

        # 4. New Target Selection (Chỉ khi CHƯA có mục tiêu hoặc mục tiêu cũ đã biến mất hẳn > lock_timeout):
        if selected is None and not has_active_lock:
            cx = win_w / 2.0
            cy = win_h / 2.0
            selected = min(targets, key=lambda t: math.hypot(t['aim_x'] - cx, t['aim_y'] - cy))
            if selected is not None:
                self.locked_track = LockedZombieTrack(
                    selected['box_x'], selected['box_y'], selected['box_w'], selected['box_h'],
                    selected['aim_x'], selected['aim_y'], selected['cls'], selected['track_id']
                )
                self.tracking_state = TrackingState.TRACKING
        elif selected is not None and self.locked_track is not None:
            self.locked_track.update(
                selected['box_x'], selected['box_y'], selected['box_w'], selected['box_h'],
                selected['aim_x'], selected['aim_y'], selected['track_id'], now, selected['cls']
            )
            self.tracking_state = TrackingState.TRACKING

        if self.locked_track is None or selected is None:
            return None

        self.locked_pos = (self.locked_track.aim_x, self.locked_track.aim_y)
        self.locked_track_id = self.locked_track.track_id
        self.locked_w = self.locked_track.box_w
        self.locked_h = self.locked_track.box_h
        self.locked_cls = self.locked_track.cls_id
        self.locked_time = self.locked_track.last_time

        return Target(
            x=self.locked_track.box_x,
            y=self.locked_track.box_y,
            w=self.locked_track.box_w,
            h=self.locked_track.box_h,
            cls=self.locked_track.cls_id,
            track_id=self.locked_track.track_id,
            aim_x=self.locked_track.aim_x,
            aim_y=self.locked_track.aim_y,
            vel_x=self.locked_track.vel_x,
            vel_y=self.locked_track.vel_y
        )

    def _get_active_candidate_idxs(self, classes_tensor):
        if hotkeys_watcher.clss is None:
            hotkeys_watcher.active_classes()

        active_class_ids = hotkeys_watcher.clss or []
        if not active_class_ids:
            return torch.empty(0, dtype=torch.long, device=classes_tensor.device)

        active_mask = torch.zeros_like(classes_tensor, dtype=torch.bool)
        for class_id in active_class_ids:
            active_mask |= classes_tensor == int(class_id)

        return torch.nonzero(active_mask, as_tuple=False).flatten()

    def _get_center(self, device):
        key = (capture.screen_x_center, capture.screen_y_center, str(device))
        if self._center is None or self._center_cache_key != key:
            self._center = torch.tensor(
                [capture.screen_x_center, capture.screen_y_center],
                dtype=torch.float32,
                device=device
            )
            self._center_cache_key = key
        return self._center

    def update_settings(self):
        self.arch = self.get_arch()
        self._center = None
        self._center_cache_key = None
        self.locked_track = None
        self.locked_pos = None
        self.locked_track_id = None
        self.locked_w = 0.0
        self.locked_h = 0.0
        self.locked_cls = None
        self.locked_time = 0.0
        self.last_aim_time = 0.0

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
        else:
            return f'cuda:{cfg.AI_device}'


frameParser = FrameParser()
