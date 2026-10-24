from __future__ import annotations

import os
import sys
import time

# Ensure CUDA runtime DLLs (cublas, cudart) in torch/lib are discoverable by ONNX Runtime on Windows
torch_lib = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

from ultralytics import YOLO
import torch
import supervision as sv
from trackers import ByteTrackTracker

from logic.config_watcher import cfg
from logic.checks import run_checks
from logic.logger import logger

_cached_predict_kwargs = None
_last_cfg_snapshot = None

def _get_predict_kwargs(tracker):
    global _cached_predict_kwargs, _last_cfg_snapshot
    cfg_snapshot = (
        cfg.ai_model_image_size,
        cfg.AI_conf,
        cfg.AI_device,
        tracker is not None
    )
    if _cached_predict_kwargs is None or cfg_snapshot != _last_cfg_snapshot:
        ai_device = str(cfg.AI_device).lower()
        _cached_predict_kwargs = dict(
            imgsz=cfg.ai_model_image_size,
            conf=cfg.AI_conf,
            iou=0.50,
            device=cfg.AI_device,
            quantize=16 if "cpu" not in ai_device else None,
            max_det=20,
            agnostic_nms=False,
            augment=False,
            vid_stride=1,
            visualize=False,
            verbose=False,
            show_boxes=False,
            show_labels=False,
            show_conf=False,
            save=False,
            show=False,
            stream=True,
            cfg="logic/tracker.yaml" if tracker else "logic/game.yaml"
        )
        _last_cfg_snapshot = cfg_snapshot
    return _cached_predict_kwargs

@torch.inference_mode()
def perform_detection(model, image, tracker: ByteTrackTracker | None = None):
    kwargs = _get_predict_kwargs(tracker)
    results = model.predict(source=image, **kwargs)

    if tracker:
        for res in results:
            det = sv.Detections.from_ultralytics(res)
            return tracker.update(det)
    else:
        if isinstance(results, list):
            return results[0] if results else None
        return next(results, None)

def init():
    run_checks()
    if torch.cuda.is_available():
        # RTX 2050 Ampere (Compute 8.6) Tensor Cores & cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            # Cap PyTorch memory fraction to 25% (~1GB) so the game has 3GB+ dedicated VRAM
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
        except Exception:
            pass
        torch.cuda.empty_cache()

    from logic.capture import capture
    from logic.visual import visuals
    from logic.frame_parser import frameParser
    from logic.hotkeys_watcher import hotkeys_watcher
    from logic.shooting import shooting
    from logic.mouse import mouse
    from logic.platform import input_backend

    tracker = ByteTrackTracker() if not cfg.disable_tracker else None
    
    try:
        model = YOLO(f"models/{cfg.AI_model_name}", task="detect")
    except Exception as e:
        logger.error(f"An error occurred when loading the AI model:\n{e}")
        raise SystemExit(0)
        
    idle_frame_counter = 0

    while True:
        image = capture.get_new_frame()

        if image is None:
            continue

        if cfg.circle_capture:
            image = capture.convert_to_circle(image)

        if hotkeys_watcher.app_pause != 0:
            visuals.clear()
            shooting.shoot(False, False)
            if cfg.show_window or cfg.show_overlay:
                visuals.submit_frame(image)
            continue

        # Smart On-Demand GPU Compute:
        # Check if the player is actively aiming, firing, or locked onto a target
        now = time.time()
        is_aiming = mouse.get_shooting_key_state() or cfg.mouse_auto_aim
        is_firing = shooting.button_pressed or input_backend.is_pressed("LeftMouseButton")
        combat_active = is_aiming or is_firing or ((now - frameParser.last_aim_time <= 0.40) and frameParser.locked_pos is not None) or ((now - frameParser.locked_time <= 0.40) and frameParser.locked_pos is not None)

        if not combat_active:
            # Player is not in combat (walking, looting, exploring):
            if cfg.show_window or cfg.show_overlay:
                idle_frame_counter += 1
                # In preview mode, run inference at a low rate (1 out of 4 frames, ~10 FPS)
                # to keep preview responsive while saving 75% GPU power!
                if idle_frame_counter % 4 == 0:
                    result = perform_detection(model, image, tracker)
                    frameParser.parse(result)
                else:
                    visuals.clear()
                visuals.submit_frame(image)
            else:
                # Production gaming mode (show_window = False):
                # Skip GPU inference entirely! Sleep 12ms to yield 100% GPU and CPU resources to the game!
                # The game runs with silky-smooth native 100+ FPS!
                time.sleep(0.012)
            continue

        # Active combat: Run at full speed!
        idle_frame_counter = 0
        result = perform_detection(model, image, tracker)
        frameParser.parse(result)

        if cfg.show_window or cfg.show_overlay:
            visuals.submit_frame(image)

if __name__ == "__main__":
    init()
