from __future__ import annotations

import configparser
import os
import threading

from logic.config_watcher import cfg
from logic.logger import logger
from logic.platform import IS_WINDOWS

if IS_WINDOWS:
    import winsound


class WeaponProfile:
    def __init__(
        self,
        key: str,
        name: str,
        recoil_pitch: float,
        recoil_spray_scale: float,
        recoil_max_pitch: float,
        hotkey: str = "None",
        beep_freq: int = 1000
    ):
        self.key = key
        self.name = name
        self.recoil_pitch = float(recoil_pitch)
        self.recoil_spray_scale = float(recoil_spray_scale)
        self.recoil_max_pitch = float(recoil_max_pitch)
        self.hotkey = hotkey
        self.beep_freq = int(beep_freq)


class WeaponManager:
    def __init__(self):
        self.profiles: dict[str, WeaponProfile] = {}
        self.current_key: str = "ak12"
        self.lock = threading.Lock()
        self.load_profiles()

    def load_profiles(self):
        with self.lock:
            self.profiles.clear()
            config_path = "config.ini"
            if not os.path.exists(config_path):
                return

            parser = configparser.ConfigParser()
            parser.read(config_path, encoding="utf-8")

            # Default profile fallbacks
            self.profiles["ak12"] = WeaponProfile(
                key="ak12",
                name="AK-12",
                recoil_pitch=2.4,
                recoil_spray_scale=0.40,
                recoil_max_pitch=4.8,
                hotkey="F5",
                beep_freq=1200
            )
            self.profiles["default"] = WeaponProfile(
                key="default",
                name="Standard / Multi-Gun",
                recoil_pitch=2.8,
                recoil_spray_scale=0.60,
                recoil_max_pitch=6.5,
                hotkey="F6",
                beep_freq=800
            )

            # Load any [Weapon_*] sections from config.ini
            for section in parser.sections():
                if section.lower().startswith("weapon_"):
                    key = section[7:].lower()
                    name = parser.get(section, "name", fallback=key.upper())
                    pitch = parser.getfloat(section, "recoil_pitch", fallback=cfg.recoil_pitch)
                    scale = parser.getfloat(section, "recoil_spray_scale", fallback=cfg.recoil_spray_scale)
                    max_p = parser.getfloat(section, "recoil_max_pitch", fallback=cfg.recoil_max_pitch)
                    hotkey = parser.get(section, "hotkey", fallback="None")
                    beep = parser.getint(section, "beep_freq", fallback=1000)

                    self.profiles[key] = WeaponProfile(
                        key=key,
                        name=name,
                        recoil_pitch=pitch,
                        recoil_spray_scale=scale,
                        recoil_max_pitch=max_p,
                        hotkey=hotkey,
                        beep_freq=beep
                    )

            cur = "ak12"
            if parser.has_section("Weapons"):
                cur = parser.get("Weapons", "current_weapon", fallback="ak12").lower()

            if cur in self.profiles:
                self.apply_profile(cur, play_sound=False)

    def apply_profile(self, key: str, play_sound: bool = True):
        if key not in self.profiles:
            return

        profile = self.profiles[key]
        self.current_key = key

        # Dynamically inject into runtime config for mouse.py
        cfg.recoil_pitch = profile.recoil_pitch
        cfg.recoil_spray_scale = profile.recoil_spray_scale
        cfg.recoil_max_pitch = profile.recoil_max_pitch

        logger.info(f"[Weapons] Switched to profile: {profile.name} (Pitch: {profile.recoil_pitch}, Scale: {profile.recoil_spray_scale}, Max: {profile.recoil_max_pitch})")

        if play_sound and IS_WINDOWS:
            threading.Thread(target=self._play_beep, args=(profile.beep_freq,), daemon=True).start()

    def _play_beep(self, freq: int):
        try:
            winsound.Beep(freq, 120)
        except Exception:
            pass

    def check_hotkeys(self, input_backend, prev_pressed: set) -> set:
        current_pressed = set()
        for key, profile in self.profiles.items():
            if profile.hotkey and profile.hotkey != "None":
                if input_backend.is_pressed(profile.hotkey):
                    current_pressed.add(profile.hotkey)
                    if profile.hotkey not in prev_pressed:
                        self.apply_profile(key, play_sound=True)
        return current_pressed


weapon_manager = WeaponManager()
