import cv2


SPEED_OPTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED_INDEX = SPEED_OPTIONS.index(1.0)
PROGRESS_TRACKBAR = "Progress"
SPEED_TRACKBAR = "Speed 0.25 0.5 1 2 4x"


class VideoControls:
    def __init__(self, window_name, total_frames, fps, start_frame=0):
        self.window_name = window_name
        self.total_frames = max(1, int(total_frames))
        self.fps = max(1.0, float(fps))
        self.current_frame = max(0, min(int(start_frame), self.total_frames - 1))
        self.speed_index = DEFAULT_SPEED_INDEX
        self.paused = False
        self.quit_requested = False
        self._requested_frame = None
        self._syncing_progress = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            PROGRESS_TRACKBAR,
            self.window_name,
            self.current_frame,
            max(1, self.total_frames - 1),
            self._on_progress,
        )
        cv2.createTrackbar(
            SPEED_TRACKBAR,
            self.window_name,
            self.speed_index,
            len(SPEED_OPTIONS) - 1,
            self._on_speed,
        )
        self._syncing_progress = False

    @property
    def speed(self):
        return SPEED_OPTIONS[self.speed_index]

    @property
    def frame_step(self):
        return max(1, int(self.speed))

    @property
    def has_pending_seek(self):
        return self._requested_frame is not None

    def sync_progress(self, frame_number):
        self.current_frame = max(0, min(int(frame_number), self.total_frames - 1))
        self._syncing_progress = True
        cv2.setTrackbarPos(PROGRESS_TRACKBAR, self.window_name, self.current_frame)
        self._syncing_progress = False

    def consume_seek(self):
        requested_frame = self._requested_frame
        self._requested_frame = None
        return requested_frame

    def wait(self, execution_time_ms, target_frame_time_ms):
        delay = max(
            1,
            int(target_frame_time_ms / self.speed - execution_time_ms),
        )
        self._handle_key(cv2.waitKey(delay) & 0xFF)
        while self.paused and not self.quit_requested and not self.has_pending_seek:
            self._handle_key(cv2.waitKey(50) & 0xFF)
        return self.quit_requested

    def status_text(self):
        current = format_timestamp(self.current_frame / self.fps)
        total = format_timestamp((self.total_frames - 1) / self.fps)
        state = "Paused" if self.paused else "Playing"
        return f"{current} / {total}  {self.speed:g}x  {state}"

    def _on_progress(self, frame_number):
        if not self._syncing_progress:
            self._requested_frame = int(frame_number)

    def _on_speed(self, speed_index):
        self.speed_index = int(speed_index)

    def _handle_key(self, key):
        if key == 255:
            return
        if key in (ord("q"), 27):
            self.quit_requested = True
        elif key == ord(" "):
            self.paused = not self.paused
        elif key in (ord("a"), 81):
            self._request_relative_seek(-5)
        elif key in (ord("d"), 83):
            self._request_relative_seek(5)
        elif key in (ord("["), ord(",")):
            self._set_speed_index(self.speed_index - 1)
        elif key in (ord("]"), ord(".")):
            self._set_speed_index(self.speed_index + 1)

    def _request_relative_seek(self, seconds):
        target = self.current_frame + int(seconds * self.fps)
        self._requested_frame = max(0, min(target, self.total_frames - 1))

    def _set_speed_index(self, speed_index):
        self.speed_index = max(0, min(int(speed_index), len(SPEED_OPTIONS) - 1))
        cv2.setTrackbarPos(SPEED_TRACKBAR, self.window_name, self.speed_index)


def draw_playback_status(frame, status_text):
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, frame.shape[0] - 48),
        (390, frame.shape[0] - 6),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        status_text,
        (20, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
