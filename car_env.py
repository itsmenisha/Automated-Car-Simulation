import cv2
import numpy as np


class CarEnv:
    def __init__(self, render=True):
        # --- Load assets ---
        self.track = cv2.imread("assets/track.png")
        if self.track is None:
            raise RuntimeError("Track image not found at assets/track.png")
        self.track = cv2.resize(self.track, (600, 400))

        self.car_img = cv2.imread("assets/car.png", cv2.IMREAD_UNCHANGED)
        if self.car_img is None:
            raise RuntimeError("Car image not found at assets/car.png")
        self.car_img = cv2.resize(self.car_img, (5, 15))

        # Track 3
        # # --- Track setup ---
        self.track_width = 40
        self.start_x, self.start_y, self.start_angle = 530.0, 200.0, 90.0
        self.finish_line = {"xmin": 520, "xmax": 560, "ymin": 230, "ymax": 250}
        self.total_track_distance = 300.0

        # # for track1
        # # --- Track setup ---
        # self.track_width = 40
        # self.start_x, self.start_y, self.start_angle = 450.0, 200.0, 90.0
        # self.finish_line = {"xmin": 440, "xmax": 480, "ymin": 230, "ymax": 250}
        # self.total_track_distance = 300.0

        # for track2
        # --- Track setup ---
        # self.track_width = 40
        # self.start_x, self.start_y, self.start_angle = 480.0, 180.0, 90.0
        # self.finish_line = {"xmin": 460, "xmax": 500, "ymin": 220, "ymax": 240}
        # self.total_track_distance = 300.0

        # Rendering
        self.render_enabled = render
        self._window_setup_done = False
        self.milestone_rewards = [False] * 10

        self.current_generation = 0
        self.max_distance_achieved = 0.0
        self.visited_finish_line = False
        self.finish_distance = 0.0

        # Reset state
        self.reset()

    def reset(self):
        self.car_pos = [self.start_x, self.start_y]
        self.car_angle = self.start_angle
        self.speed = 8.0
        self.distance_traveled = 0.0
        self.done = False

        self.milestone_rewards = [False] * 10
        self.prev_angle = self.car_angle
        self.prev_speed = self.speed

        # <<< CRITICAL CHECKPOINT RESET >>>
        self.passed_checkpoint = False
        self.lap_completed = False

        self.visited_far_zone = False
        self.max_far_distance = 0.0
        self.returned_from_far = False

        self.stagnation_steps = 0

        # <<< NEW: Max distance must be reset >>>
        self.max_distance_achieved = 0.0

        return self.get_state()

    def get_state(self):
        sensors = np.array(
            [self.sensor(a) for a in [-90, -60, -30, 0, 30, 60, 90]], dtype=np.float32)
        angle_rad = np.radians(self.car_angle)
        angle_features = [np.sin(angle_rad), np.cos(angle_rad)]
        return np.append(sensors, angle_features)

    def sensor(self, angle_offset):
        angle = self.car_angle + angle_offset
        for distance in range(1, 100, 3):
            x = int(self.car_pos[0] + distance * np.cos(np.radians(angle)))
            y = int(self.car_pos[1] - distance * np.sin(np.radians(angle)))
            if x < 0 or x >= 600 or y < 0 or y >= 400:
                return distance / 100.0
            if np.sum(self.track[y, x]) > 700:
                return distance / 100.0
        return 1.0

    def check_collision(self):
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        if x < 0 or x >= 600 or y < 0 or y >= 400:
            return True
        points = [(x, y), (x - 5, y), (x + 5, y), (x, y - 5), (x, y + 5)]
        for px, py in points:
            if px < 0 or px >= 600 or py < 0 or py >= 400:
                return True
            if np.sum(self.track[py, px]) > 700:
                return True
        return False

    def check_finish_line(self):
        x, y = self.car_pos

        if self.distance_traveled < 50.0:
            return False

        inside = (
            self.finish_line["xmin"] <= x <= self.finish_line["xmax"] and
            self.finish_line["ymin"] <= y <= self.finish_line["ymax"]
        )

        # <<< MODIFIED: Must be inside the finish line AND have passed the checkpoint >>>
        if not inside or not self.passed_checkpoint:
            return False

        if not (80 <= self.car_angle <= 100):
            return False

        self.lap_completed = True
        return True

    def step_discrete(self, action_idx):
        if self.done:
            return self.get_state(), 0.0, True

        # --- Actions ---
        steering_delta, speed_delta = 0.0, 0.0
        if action_idx == 0:
            steering_delta = 15.0
        elif action_idx == 1:
            steering_delta = -15.0
        elif action_idx == 2:
            speed_delta = 2.0
        elif action_idx == 3:
            speed_delta = -2.0

        # --- Update dynamics ---
        self.speed = float(np.clip(self.speed + speed_delta, 2.0, 12.0))
        self.car_angle = (self.car_angle + steering_delta) % 360.0
        movement = self.speed

        old_pos = self.car_pos.copy()
        self.car_pos[0] += movement * np.cos(np.radians(self.car_angle))
        self.car_pos[1] -= movement * np.sin(np.radians(self.car_angle))
        self.distance_traveled += movement

        # <<< NEW: Update max distance achieved >>>
        if self.distance_traveled > self.max_distance_achieved:
            self.max_distance_achieved = self.distance_traveled

        # --- Common values ---
        sensors = self.get_state()[:7]
        avg_distance_to_walls = float(np.mean(sensors))
        min_distance_to_wall = float(np.min(sensors))

        finish_cx = (self.finish_line["xmin"] + self.finish_line["xmax"]) / 2.0
        finish_cy = (self.finish_line["ymin"] + self.finish_line["ymax"]) / 2.0
        old_dist = float(np.linalg.norm(
            [old_pos[0] - finish_cx, old_pos[1] - finish_cy]))
        new_dist = float(np.linalg.norm(
            [self.car_pos[0] - finish_cx, self.car_pos[1] - finish_cy]))

        angle_change = abs(
            (self.car_angle - self.prev_angle + 540.0) % 360.0 - 180.0)
        prev_speed = self.prev_speed
        self.prev_angle = self.car_angle
        self.prev_speed = self.speed

        x, y = self.car_pos
        gen = getattr(self, "current_generation", 0)

        # <<< NEW: Checkpoint Tracking >>>
        checkpoint_zone = {"xmin": 20, "xmax": 40, "ymin": 180, "ymax": 220}
        if checkpoint_zone["xmin"] <= x <= checkpoint_zone["xmax"] and \
                checkpoint_zone["ymin"] <= y <= checkpoint_zone["ymax"]:
            self.passed_checkpoint = True
        # *****************************************************************

        # --- Reward ---
        reward = 0.0
        reward += 0.05 	# baseline survival reward

        # --- NEW: UNIFIED PROGRESS REWARD (Strong incentive to follow track) ---
        milestones = [i / 10.0 for i in range(1, 11)]
        track_progress_ratio = self.distance_traveled / \
            max(self.total_track_distance, 1.0)
        current_milestone_reward = 0.0

        for i, m in enumerate(milestones):
            if track_progress_ratio >= m and not self.milestone_rewards[i]:
                current_milestone_reward += 200.0  # High reward for 10% progress
                self.milestone_rewards[i] = True

        reward += current_milestone_reward

        # --- CHECK FOR INVALID EXTREME FITNESS (LOOP DETECTION) ---

        # --- END UNIFIED PROGRESS REWARD (Replaces old milestone blocks) ---
        # --- Finish Line Logic (like far zone) ---
        finish_zone = self.finish_line  # assuming you already defined self.finish_line
        finish_cx = (finish_zone["xmin"] + finish_zone["xmax"]) / 2.0
        finish_cy = (finish_zone["ymin"] + finish_zone["ymax"]) / 2.0
        finish_dist = float(np.linalg.norm(
            [self.car_pos[0] - finish_cx, self.car_pos[1] - finish_cy]))

        # Reward when reaching the finish zone for the first time
        if (
            not self.visited_finish_line
            and finish_zone["xmin"] <= x <= finish_zone["xmax"]
            and finish_zone["ymin"] <= y <= finish_zone["ymax"]
        ):
            reward += 80000.0  # one-time finish line bonus
            self.visited_finish_line = True
            self.finish_distance = self.distance_traveled
            self.done = True  # end the episode
            print(
                f"[FINISH] Reached finish line at distance {self.finish_distance:.2f}.")

        # --- Far zone logic ---
        far_zone = {"xmin": 20, "xmax": 40, "ymin": 120, "ymax": 130}
        far_cx = (far_zone["xmin"] + far_zone["xmax"]) / 2.0
        far_cy = (far_zone["ymin"] + far_zone["ymax"]) / 2.0
        far_dist = float(np.linalg.norm(
            [self.car_pos[0] - far_cx, self.car_pos[1] - far_cy]))

        if not self.visited_far_zone and far_zone["xmin"] <= x <= far_zone["xmax"] and far_zone["ymin"] <= y <= far_zone["ymax"]:
            reward += 80000
            self.visited_far_zone = True
            self.max_far_distance = far_dist

        if (
            not getattr(self, "visited_finish_line", False)
            and not getattr(self, "visited_far_zone", False)
            and self.distance_traveled >= 10000
        ):
            reward -= 1_000_000.0
            self.done = True
            print(
                f"[LOOP DETECTED] Car reached {self.distance_traveled:.2f} units without finishing or far zone — penalized.")

        # REMOVED: Exploitable continuous 'far_dist' reward has been deleted.

        if self.visited_far_zone and not self.returned_from_far and new_dist < old_dist and new_dist < 0.8 * old_dist:
            reward += 10000
            self.returned_from_far = True

        # --- Stagnation penalty ---
        dist_diff = old_dist - new_dist   # positive if moved closer
        angle_diff = angle_change         # from your existing computation

        if not hasattr(self, "rotation_changes"):
            self.rotation_changes = 0
        if not hasattr(self, "stagnation_steps"):
            self.stagnation_steps = 0

        if angle_diff > 10:
            self.rotation_changes += 1
        else:
            self.rotation_changes = max(
                0, self.rotation_changes - 1)

        if dist_diff > 5 or self.speed > 1.0:
            self.stagnation_steps = 0
            self.rotation_changes = 0
        else:
            self.stagnation_steps += 1

        if self.rotation_changes > 500 and self.stagnation_steps > 5:
            reward -= 20000000
            self.done = True
            print(
                f"[STAGNATION] Car spun in place: {self.rotation_changes} rotations, no progress.")

        if gen < 20:

            reward += 0.9 * movement

            reward += 5 * self.max_distance_achieved

        elif gen < 40:

            reward += 0.1 * movement
            reward += avg_distance_to_walls * 0.3
            reward += min_distance_to_wall * 0.2
            reward -= 0.02 * angle_change
            if angle_change < 5.0:
                reward += 0.2

            reward += 0.05 * self.max_distance_achieved

        elif gen < 60:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement
            reward += avg_distance_to_walls * 0.3
            reward += min_distance_to_wall * 0.2
            reward -= 0.02 * angle_change
            if angle_change < 5.0:
                reward += 0.2
            reward += 0.1 * self.speed
            reward += 0.05 * max(self.speed - prev_speed, 0.0)
            if self.speed < 3.0:
                reward -= 1.0

        elif gen < 70:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement

        elif gen < 80:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement

        elif gen < 90:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement
            reward -= 0.02 * angle_change
            if angle_change < 5.0:
                reward += 0.2

        elif gen < 100:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement

        else:
            reward += (old_dist - new_dist) * 2.0
            reward += 0.1 * movement
            reward += avg_distance_to_walls * 0.3
            reward += min_distance_to_wall * 0.2
            reward += 0.1 * self.speed
            reward += 0.05 * max(self.speed - prev_speed, 0.0)

        if self.check_collision():
            reward -= 30.0 + 10.0 * self.speed
            self.done = True

        # --- Finish line bonus ---
        if self.check_finish_line():
            reward += 800.0
            self.laps_completed += 1
            self.done = True

        return self.get_state(), reward, self.done

    def render(self, all_cars=None):
        if not self.render_enabled:
            return 1  # Headless mode

        if not self._window_setup_done:
            cv2.namedWindow("Car Simulator", cv2.WINDOW_NORMAL)
            self._window_setup_done = True

        display = self.track.copy()

        # Start area
        cv2.rectangle(
            display,
            (int(self.start_x - 10), int(self.start_y - self.track_width / 2)),
            (int(self.start_x + 10), int(self.start_y + self.track_width / 2)),
            (255, 0, 0), 2
        )

        # Finish line
        cv2.rectangle(
            display,
            (self.finish_line["xmin"], self.finish_line["ymin"]),
            (self.finish_line["xmax"], self.finish_line["ymax"]),
            (0, 255, 0), 2
        )

        # Checkpoint zone
        checkpoint_zone = {"xmin": 20, "xmax": 40, "ymin": 180, "ymax": 220}
        cv2.rectangle(
            display,
            (checkpoint_zone["xmin"], checkpoint_zone["ymin"]),
            (checkpoint_zone["xmax"], checkpoint_zone["ymax"]),
            (255, 255, 0), 2  # Cyan
        )

        # Far zone
        far_zone = {"xmin": 20, "xmax": 40, "ymin": 120, "ymax": 130}
        cv2.rectangle(
            display,
            (far_zone["xmin"], far_zone["ymin"]),
            (far_zone["xmax"], far_zone["ymax"]),
            (0, 255, 255), 2  # Yellow
        )

        # Draw cars
        cars_to_draw = all_cars if all_cars is not None else [
            (self.car_pos, self.car_angle)]
        for pos, angle in cars_to_draw:
            M = cv2.getRotationMatrix2D(
                (self.car_img.shape[1] // 2,
                 self.car_img.shape[0] // 2), -angle, 1.0
            )
            rotated = cv2.warpAffine(
                self.car_img, M,
                (self.car_img.shape[1], self.car_img.shape[0]),
                borderMode=cv2.BORDER_TRANSPARENT
            )
            x = int(pos[0] - rotated.shape[1] // 2)
            y = int(pos[1] - rotated.shape[0] // 2)
            self.overlay_image(display, rotated, x, y)

        cv2.imshow("Car Simulator", display)
        return cv2.waitKey(1)

    @staticmethod
    def overlay_image(bg, overlay, x, y):
        h, w = overlay.shape[:2]

        # Valid draw area
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])

        overlay_x1 = max(0, -x)
        overlay_y1 = max(0, -y)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        if x1 >= x2 or y1 >= y2:
            return

        # Alpha blending
        if overlay.shape[2] == 4:
            alpha = overlay[overlay_y1:overlay_y2,
                            overlay_x1:overlay_x2, 3] / 255.0
        else:
            alpha = np.ones((y2 - y1, x2 - x1), dtype=np.float32)

        for c in range(3):
            bg[y1:y2, x1:x2, c] = (
                alpha * overlay[overlay_y1:overlay_y2,
                                overlay_x1:overlay_x2, c]
                + (1 - alpha) * bg[y1:y2, x1:x2, c]
            )
