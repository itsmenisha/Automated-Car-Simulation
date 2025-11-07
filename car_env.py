import cv2
import numpy as np


class CarEnv:
    def __init__(self, render=True):
        # --- 1. Load Images ---
        self.track = cv2.imread("assets/track.png")
        if self.track is None:
            raise RuntimeError("Track image not found at assets/track1.png")
        self.track = cv2.resize(self.track, (600, 400))

        self.car_img = cv2.imread("assets/car.png", cv2.IMREAD_UNCHANGED)
        if self.car_img is None:
            raise RuntimeError("Car image not found at assets/car.png")
        self.car_img = cv2.resize(self.car_img, (15, 40))

        # for easier track
        # Track properties
        # self.track_width = 40
        # self.start_x, self.start_y, self.start_angle = 465.0, 210.0, 120.0
        # self.finish_line = {
        #     "xmin": 450, "xmax": 480,
        #     "ymin": 240, "ymax": 250
        # }

        # Track properties
        self.track_width = 40
        self.start_x, self.start_y, self.start_angle = 535.0, 200.0, 120.0
        self.finish_line = {
            "xmin": 520, "xmax": 560,
            "ymin": 230, "ymax": 250
        }

        self.min_distance_for_lap = 50
        self.total_track_distance = 1000.0

        # Rendering flag
        self.render_enabled = render
        self._window_setup_done = False

        # Reset car state
        self.reset()

    def reset(self):
        self.car_pos = [self.start_x, self.start_y]
        self.car_angle = self.start_angle
        self.speed = 8.0
        self.distance_traveled = 0.0
        self.laps_completed = 0
        self.done = False
        self.milestone_rewards = [False, False, False]
        self.prev_angle = self.car_angle
        return self.get_state()

    # --- Sensors and State ---
    def get_state(self):
        # 7 sensors: [-90, -60, -30, 0, 30, 60, 90]
        sensors = np.array([self.sensor(a)
                           for a in [-90, -60, -30, 0, 30, 60, 90]])
        angle_norm = (self.car_angle % 360.0) / 360.0
        return np.append(sensors, angle_norm)

    def sensor(self, angle_offset):
        angle = self.car_angle + angle_offset
        for distance in range(1, 100):
            x = int(self.car_pos[0] + distance * np.cos(np.radians(angle)))
            y = int(self.car_pos[1] - distance * np.sin(np.radians(angle)))
            if x < 0 or x >= 600 or y < 0 or y >= 400:
                return distance / 100.0
            pixel = self.track[y, x]
            if np.sum(pixel) > 700:  # Treat bright pixels as walls
                return distance / 100.0
        return 1.0

    # --- Collision ---
    def check_collision(self):
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        if x < 0 or x >= 600 or y < 0 or y >= 400:
            return True

        points = [(x, y), (x-5, y), (x+5, y), (x, y-5), (x, y+5)]
        for px, py in points:
            if px < 0 or px >= 600 or py < 0 or py >= 400:
                return True
            if np.sum(self.track[py, px]) > 700:
                return True
        return False

    def step_discrete(self, action_idx):
        if self.done:
            return self.get_state(), 0.0, True

        # --- 1. Map discrete actions ---
        steering_delta, speed_delta = 0, 0
        if action_idx == 0:
            steering_delta = 15.0
        elif action_idx == 1:
            steering_delta = -15.0
        elif action_idx == 2:
            speed_delta = 2.0
        elif action_idx == 3:
            speed_delta = -2.0

        # --- 2. Update car state ---
        self.speed = np.clip(self.speed + speed_delta, 2.0, 12.0)
        self.car_angle += steering_delta
        movement = self.speed

        old_pos = self.car_pos.copy()
        self.car_pos[0] += movement * np.cos(np.radians(self.car_angle))
        self.car_pos[1] -= movement * np.sin(np.radians(self.car_angle))
        self.distance_traveled += movement

        # --- 3. Initialize reward ---
        reward = 0.0

        # Progress toward finish line
        finish_cx = (self.finish_line["xmin"] + self.finish_line["xmax"]) / 2
        finish_cy = (self.finish_line["ymin"] + self.finish_line["ymax"]) / 2
        old_dist = np.linalg.norm([old_pos[0]-finish_cx, old_pos[1]-finish_cy])
        new_dist = np.linalg.norm(
            [self.car_pos[0]-finish_cx, self.car_pos[1]-finish_cy])
        reward += (old_dist - new_dist) * 2.0  # Encourage forward movement

        # --- 4. Safety / collision penalty ---
        if self.check_collision():
            reward -= 100.0  # Heavy penalty for crashing
            self.done = True
            return self.get_state(), reward, self.done

        # --- 5. Sensor / center-of-track reward ---
        sensors = self.get_state()[:7]  # 7 distance sensors
        avg_distance_to_walls = np.mean(sensors)
        reward += avg_distance_to_walls * 0.5  # Encourage staying in track center

        # --- 6. Steering smoothness reward ---
        if not hasattr(self, "prev_angle"):
            self.prev_angle = self.car_angle
        # Penalize sharp turns
        reward -= 0.05 * abs(self.car_angle - self.prev_angle)
        self.prev_angle = self.car_angle

        # --- 7. Speed reward ---
        reward += 0.2 * self.speed  # Encourage higher speeds
        if hasattr(self, "prev_speed"):
            # Bonus for accelerating
            reward += 0.1 * max(self.speed - self.prev_speed, 0)
        self.prev_speed = self.speed

        # --- 8. Milestones ---
        if not hasattr(self, "milestone_rewards"):
            self.milestone_rewards = [False, False, False]  # 3 milestones
        milestones = [0.33, 0.66, 1.0]
        for i, m in enumerate(milestones):
            if self.distance_traveled >= m * self.total_track_distance and not self.milestone_rewards[i]:
                if self.speed >= 8.0:  # Bonus for fast milestone crossing
                    reward += 100.0
                    self.milestone_rewards[i] = True

        # --- 9. Finish line reward ---
        if self.check_finish_line():
            reward += 1000.0
            self.laps_completed += 1
            self.done = True
        return self.get_state(), reward, self.done

    def check_finish_line(self):
        x, y = self.car_pos
        if self.distance_traveled < self.min_distance_for_lap:
            return False
        return (self.finish_line["xmin"] <= x <= self.finish_line["xmax"] and
                self.finish_line["ymin"] <= y <= self.finish_line["ymax"])

    # --- Rendering ---
    def render(self, all_cars=None):
        if not self.render_enabled:
            return  # Skip rendering for headless training

        if not self._window_setup_done:
            cv2.namedWindow("Car Simulator", cv2.WINDOW_NORMAL)
            self._window_setup_done = True

        display = self.track.copy()

        # Draw start/finish
        cv2.rectangle(display,
                      (int(self.start_x-10), int(self.start_y-self.track_width/2)),
                      (int(self.start_x+10), int(self.start_y+self.track_width/2)),
                      (255, 0, 0), 2)
        cv2.rectangle(display,
                      (self.finish_line["xmin"], self.finish_line["ymin"]),
                      (self.finish_line["xmax"], self.finish_line["ymax"]),
                      (0, 255, 0), 2)

        # Draw cars
        cars_to_draw = all_cars if all_cars is not None else [
            (self.car_pos, self.car_angle)]
        for pos, angle in cars_to_draw:
            M = cv2.getRotationMatrix2D(
                (self.car_img.shape[1]//2, self.car_img.shape[0]//2), -angle, 1.0)
            rotated = cv2.warpAffine(
                self.car_img, M, (self.car_img.shape[1], self.car_img.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
            x = int(pos[0]-rotated.shape[1]//2)
            y = int(pos[1]-rotated.shape[0]//2)
            self.overlay_image(display, rotated, x, y)

        cv2.imshow("Car Simulator", display)
        return cv2.waitKey(1)

    @staticmethod
    def overlay_image(bg, overlay, x, y):
        h, w = overlay.shape[:2]
        if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
            return
        alpha = overlay[:, :, 3] / \
            255.0 if overlay.shape[2] == 4 else np.ones((h, w))
        for c in range(3):
            bg[y:y+h, x:x+w, c] = alpha*overlay[:, :, c] + \
                (1-alpha)*bg[y:y+h, x:x+w, c]
