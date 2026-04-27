import os
import time

from locust import HttpUser, between, task


BURST_WINDOW_SECONDS = float(os.getenv("BURST_WINDOW_SECONDS", "20"))
BURST_MULTIPLIER = int(os.getenv("BURST_MULTIPLIER", "12"))


class BurstTrafficUser(HttpUser):
    # Short wait time keeps request-rate rise steep so predictive telemetry leads CPU saturation.
    wait_time = between(0.01, 0.05)

    @task(1)
    def burst_window(self):
        end_time = time.time() + BURST_WINDOW_SECONDS
        while time.time() < end_time:
            for _ in range(BURST_MULTIPLIER):
                self.client.get("/", name="BURST GET /")
