import threading
import time
from multiprocessing import Process, Manager, Event

import cv2
import keyboard
import pygetwindow as gw
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
from ultralytics import YOLO

obs_camera_url = 1


class FPS:
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0
        self.current_fps = 0

    def update(self):
        self.frames += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1:
            self.current_fps = self.frames / elapsed_time
            self.start_time = time.time()
            self.frames = 0
        return self.current_fps


model = YOLO("best.pt")


def detect_stars(queue, stop_event):
    cap = cv2.VideoCapture(obs_camera_url)

    if not cap.isOpened():
        print("Video stream not opened.")
        exit()

    fps_counter = FPS()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame_resized = cv2.resize(frame, (1920, 1080))

        frame_cropped = frame_resized[:1080 - 368, :1920 - 1519]
        results = model(frame_cropped)
        annotated_frame = results[0].plot()

        fps = fps_counter.update()
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("OBS Virtual Camera", annotated_frame)

        stars = []
        bombs = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            if label == 'star':
                stars.append(box.xyxy[0].tolist())
            elif label == 'bomb':
                bombs.append(box.xyxy[0].tolist())

        while not queue.empty():
            queue.get()

        queue.put((stars, bombs))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def click_stars(queue, stop_event):
    def check_collision(star, bombs):
        x1_star, y1_star, x2_star, y2_star = star
        for bomb in bombs:
            x1_bomb, y1_bomb, x2_bomb, y2_bomb = bomb
            if not (x2_star < x1_bomb or x1_star > x2_bomb or y2_star < y1_bomb or y1_star > y2_bomb):
                return True
        return False

    def is_star_clicked(star, clicked_stars, threshold=10):
        x1_star, y1_star, x2_star, y2_star = star
        for clicked_star in clicked_stars:
            x1_clicked, y1_clicked, x2_clicked, y2_clicked = clicked_star
            if abs(x1_star - x1_clicked) < threshold and abs(y1_star - y1_clicked) < threshold and \
                    abs(x2_star - x2_clicked) < threshold and abs(y2_star - y2_clicked) < threshold:
                return True
        return False

    mouse = MouseController()
    click_count = 0  # Initialize click counter
    clicked_stars = []  # List to track clicked stars

    while not stop_event.is_set():
        if queue.empty():
            continue

        stars, bombs = queue.get()
        if stars is None:
            break

        window = None
        windows = gw.getAllTitles()
        for title in windows:
            if "TelegramDesktop" in title:
                window = gw.getWindowsWithTitle(title)[0]
                break

        if not window:
            print("Window 'TelegramDesktop' not found.")
            return

        window.activate()

        frame_width, frame_height = 1920 - 1519, 1080 - 368
        window_width, window_height = window.width, window.height

        x_scale = window_width / frame_width
        y_scale = window_height / frame_height

        if stars:
            filtered_stars = stars
            if bombs:
                filtered_stars = [star for star in stars if not check_collision(star, bombs)]

            if filtered_stars:
                for star in filtered_stars:
                    if is_star_clicked(star, clicked_stars):
                        continue

                    clicked_stars.append(star)  # Add star to clicked list
                    if len(clicked_stars) > 50:
                        clicked_stars.pop(0)  # Remove oldest star from clicked list

                    x1, y1, x2, y2 = star
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    click_x = window.left + x_center * x_scale
                    click_y = window.top + y_center * y_scale

                    print(f"Clicking at ({click_x}, {click_y}) for star centered at ({x_center}, {y_center})")
                    mouse.position = (click_x, click_y)
                    mouse.click(Button.left, 1)

                    click_count += 1  # Increment click counter
                    print(f"Mouse clicks: {click_count}")  # Print the click count


def monitor_stop_key(stop_event):
    keyboard.wait('q')
    stop_event.set()


if __name__ == "__main__":
    manager = Manager()
    queue = manager.Queue()
    stop_event = Event()

    p1 = Process(target=detect_stars, args=(queue, stop_event))
    p2 = Process(target=click_stars, args=(queue, stop_event))

    p1.start()
    p2.start()

    stop_thread = threading.Thread(target=monitor_stop_key, args=(stop_event,))
    stop_thread.start()

    p1.join()
    queue.put(None)
    p2.join()
    stop_thread.join()
