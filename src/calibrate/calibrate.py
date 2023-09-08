""" GUI Module for calibrating web camera. Usage: $ python3 ./calibrate.py"""
import cv2
import numpy as np
from tkinter import Tk, Button, Canvas, NW
from PIL import Image, ImageTk


class CalibrateWebCam:
    """GUI model for calibrating user webcam."""
    def __init__(self, window):
        self.window = window
        self.img_height = 480
        self.img_width = 640
        self.recording = False

        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.img = None
        self.active_frame = None
        self.view_img = None

        self.canvas = Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.capture_checker_board_frame = Button(
            self.window, text="record points", command=self._capture_points
        )
        self.capture_checker_board_frame.pack(side="left")

        self.calibrate_button = Button(
            self.window, text="calibrate", command=self._on_button_calibrate
        )
        self.calibrate_button.pack(side="right")

        self.update_webcam()

        # setting chessboard size
        self.chessboard_size = (9, 6)

        # prepare object points
        self.obj_p = np.zeros((np.prod(self.chessboard_size), 3), dtype=np.float32)
        self.obj_p[:, :2] = np.mgrid[
            0: self.chessboard_size[0], 0: self.chessboard_size[1]
        ].T.reshape(-1, 2)

        # prepare recording
        self.record_min_num_frames = 20
        self._reset_recording()

    def update_webcam(self):
        rect, frame = self.video_capture.read()

        if rect:
            self.active_frame = frame
            self.img = Image.fromarray(
                cv2.cvtColor(self.active_frame, cv2.COLOR_BGR2RGB)
            )
            self.view_img = ImageTk.PhotoImage(image=self.img)
            self.canvas.create_image(0, 0, image=self.view_img, anchor=NW)
            self.window.after(15, self.update_webcam)

    def _reset_recording(self):
        self.record_cnt = 0
        self.obj_points = []
        self.img_points = []
        self.calibrate_button["state"] = "disable"

    def _capture_points(self):
        if self.active_frame is not None:
            img_gray = cv2.cvtColor(
                np.asarray(self.active_frame), cv2.COLOR_BGR2GRAY
            ).astype(np.uint8)

            ret, corners = cv2.findChessboardCorners(
                img_gray, self.chessboard_size, None
            )

            if ret:
                cv2.drawChessboardCorners(
                    self.active_frame, self.chessboard_size, corners, ret
                )

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(img_gray, corners, (9, 9), (-1, -1), criteria)

            self.obj_points.append(self.obj_p)
            self.img_points.append(corners)
            self.record_cnt += 1
            print("capture another frame, change checker board position")

        if self.record_cnt == self.record_min_num_frames:
            self.capture_checker_board_frame["state"] = "disable"
            self.calibrate_button["state"] = "active"

    def _on_button_calibrate(self):
        print("Calibrating...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            (self.img_height, self.img_width),
            None,
            None,
        )

        print("K=", K)
        print("dist=", dist)

        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], K, dist
            )
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(
                img_points2
            )
            mean_error += error
        print("mean error=", f"{mean_error} pixels")
        print("Done.")


def main():
    root = Tk()
    root.title("Calibrate Camera")
    app = CalibrateWebCam(root)
    root.mainloop()


if __name__ == "__main__":
    main()
