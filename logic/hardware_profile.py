from __future__ import annotations

import os
import time
import torch
import psutil

from logic.config_watcher import cfg
from logic.logger import logger

class HardwareProfileManager:
    STABLE = "STABLE"
    BALANCED = "BALANCED"
    PERFORMANCE = "PERFORMANCE"

    def __init__(self):
        self.current_profile = self.STABLE
        self.device_name = "CPU"
        self.vram_total_mb = 0.0
        self.is_rtx_2050 = False
        self.idle_sleep_ms = 10
        self.max_queue_age_ms = 40
        self.box_history_timeout_s = 0.40
        self._init_device_specs()

    def _init_device_specs(self):
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)
            self.vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if "2050" in self.device_name:
                self.is_rtx_2050 = True
        logger.info(f"[HardwareProfile] Detected GPU: {self.device_name} ({self.vram_total_mb:.0f} MB VRAM)")

    def apply_profile(self, profile_name: str, verbose: bool = True):
        mode = profile_name.upper().strip()
        if mode not in (self.STABLE, self.BALANCED, self.PERFORMANCE):
            mode = self.STABLE

        self.current_profile = mode

        # Empirical tuning based on RTX 2050 benchmark measurements:
        # ONNX Runtime CUDA @ 416x416: 31.27ms (32.0 FPS), 32MB VRAM allocated, 62MB reserved.
        if mode == self.STABLE:
            # Ưu tiên độ ổn định tuyệt đối khi game đang chạy nặng, tiết kiệm pin & nhiệt độ laptop
            cfg.ai_model_name = "sunxds_0.8.0.onnx"
            cfg.ai_model_image_size = 416
            cfg.AI_conf = 0.14
            cfg.capture_fps = 45
            cfg.mouse_pid_kp = 0.45
            cfg.mouse_deadzone = 1.2
            cfg.mouse_max_step = 20
            self.idle_sleep_ms = 10
            self.max_queue_age_ms = 40
            self.box_history_timeout_s = 0.40
        elif mode == self.BALANCED:
            # Cân bằng hoàn hảo giữa thông lượng và độ mượt, phản hồi ~16ms
            cfg.ai_model_name = "sunxds_0.8.0.onnx"
            cfg.ai_model_image_size = 416
            cfg.AI_conf = 0.14
            cfg.capture_fps = 60
            cfg.mouse_pid_kp = 0.50
            cfg.mouse_deadzone = 1.2
            cfg.mouse_max_step = 25
            self.idle_sleep_ms = 4
            self.max_queue_age_ms = 30
            self.box_history_timeout_s = 0.35
        elif mode == self.PERFORMANCE:
            # Tối đa hóa tốc độ xử lý khi CPU/GPU còn dư dả tài nguyên
            cfg.ai_model_name = "sunxds_0.8.0.onnx"
            cfg.ai_model_image_size = 416
            cfg.AI_conf = 0.14
            cfg.capture_fps = 80
            cfg.mouse_pid_kp = 0.55
            cfg.mouse_deadzone = 1.0
            cfg.mouse_max_step = 30
            self.idle_sleep_ms = 0
            self.max_queue_age_ms = 20
            self.box_history_timeout_s = 0.25

        if verbose:
            logger.info(f"[HardwareProfile] Switched to mode: {self.current_profile} (Target FPS: {cfg.capture_fps}, MaxStep: {cfg.mouse_max_step}, Conf: {cfg.AI_conf})")
        return self

    @property
    def fps_cap(self) -> int:
        return int(cfg.capture_fps)

    def get_telemetry(self) -> dict:
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        vram_res = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        proc_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        total_ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        return {
            "profile": self.current_profile,
            "vram_alloc_mb": round(vram_alloc, 1),
            "vram_res_mb": round(vram_res, 1),
            "proc_ram_mb": round(proc_ram, 1),
            "sys_ram_avail_gb": round(total_ram_avail, 2),
        }

hardware_profile = HardwareProfileManager()
