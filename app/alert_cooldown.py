
import time
from threading import Lock


class AlertCooldown:


    COOLDOWN_CONFIG = {

        "default_garbage": 15 * 60,
        "recyclable": 15 * 60,
        "kitchen_waste": 15 * 60,
        "hazardous": 15 * 60,
        "other": 15 * 60,

        "fire": 90,
        "smoke": 90,
    }

    def __init__(self):
        self._last_alert_time = {}  
        self._lock = Lock()

    def _get_cooldown_seconds(self, category: str) -> int:


        if category in self.COOLDOWN_CONFIG:
            return self.COOLDOWN_CONFIG[category]

        if "fire" in category.lower() or "smoke" in category.lower():
            return 90
        return self.COOLDOWN_CONFIG["default_garbage"]

    def can_alert(self, category: str) -> bool:


        with self._lock:
            now = time.time()
            last = self._last_alert_time.get(category, 0)
            cooldown = self._get_cooldown_seconds(category)
            if now - last >= cooldown:

                self._last_alert_time[category] = now
                return True
            else:
                remaining = int(cooldown - (now - last))
                print(f"[冷却抑制] 类别 '{category}' 还需等待 {remaining} 秒")
                return False

    def reset_category(self, category: str):

        with self._lock:
            self._last_alert_time.pop(category, None)


cooldown_manager = AlertCooldown()