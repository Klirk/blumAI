import threading
import tkinter as tk

import cv2
import keyboard
import mss
import numpy as np
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO


class BotApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Bot Controller")

        # Установка ширины окна
        self.root.geometry("200x100")  # Измените ширину и высоту по своему усмотрению

        self.start_button = tk.Button(root, text="Start - F5", command=self.start_bot, width=20)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop - F6", command=self.stop_bot, width=20)
        self.stop_button.pack(pady=10)

        self.running = False
        self.setup_hotkeys()
        self.window_title_start = "TelegramDesktop"
        self.window = None
        self.model = YOLO("best.pt")  # Загрузка обученной модели

    def setup_hotkeys(self):
        keyboard.add_hotkey('F5', self.start_bot)
        keyboard.add_hotkey('F6', self.stop_bot)

    def find_window(self):
        windows = gw.getAllTitles()
        print("All windows:", windows)  # Output all window titles for debugging
        for window in windows:
            if window.startswith(self.window_title_start):
                return gw.getWindowsWithTitle(window)[0]
        return None

    def capture_screen(self, bbox):
        with mss.mss() as sct:
            screenshot = sct.grab(bbox)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
            return img

    def click_on_particles(self, objects, bbox):
        # Sort objects by their y-coordinate (closer to bottom first)
        objects = sorted(objects, key=lambda obj: obj['box'][3], reverse=True)

        for obj in objects:
            name = obj['name']
            x1, y1, x2, y2 = obj['box']

            if name == 'star':
                # Calculate the center of the bounding box
                cx = bbox[0] + (x1 + x2) // 2
                cy = bbox[1] + (y1 + y2) // 2
                pyautogui.click(cx, cy, interval=0.0)
                break  # Click only on the closest star to the bottom

    def process_frame(self, img, bbox):
        results = self.model.predict(img)
        # show the result image
        cv2.imshow("Result", results[0].plot())
        cv2.waitKey(1)

        objects = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])  # Convert tensor to int
            label = results[0].names[cls_id]
            objects.append({
                'name': label,
                'box': box.xyxy[0].tolist()
            })
        self.click_on_particles(objects, bbox)

    def bot_loop(self):
        window = self.find_window()
        if window:
            top_offset = 10
            right_offset = 10
            left_offset = 10
            bottom_offset = 10
            bbox = (window.left + left_offset, window.top + top_offset, window.right - right_offset,
                    window.bottom - bottom_offset)
            while self.running:
                img = self.capture_screen(bbox)
                self.process_frame(img, bbox)

    def start_bot(self):
        if not self.running:
            self.running = True
            self.bot_thread = threading.Thread(target=self.bot_loop)
            self.bot_thread.start()

    def stop_bot(self):
        self.running = False
        if hasattr(self, 'bot_thread'):
            self.bot_thread.join()


if __name__ == "__main__":
    root = tk.Tk()
    app = BotApp(root)
    root.mainloop()
