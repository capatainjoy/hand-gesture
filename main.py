import cv2
import mediapipe as mp
import numpy as np
import math
import screen_brightness_control as sbc

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ===================== AUDIO SETUP =====================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handType in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            label = handType.classification[0].label
            lm = handLms.landmark

            x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
            x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

            cv2.circle(img, (x1, y1), 8, (255, 0, 255), -1)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            dist = distance((x1, y1), (x2, y2))

            # LEFT HAND → VOLUME
            if label == "Left":
                vol = np.interp(dist, [30, 200], [vol_min, vol_max])
                volume.SetMasterVolumeLevel(vol, None)

            # RIGHT HAND → BRIGHTNESS
            if label == "Right":
                bright = np.interp(dist, [30, 200], [0, 100])
                sbc.set_brightness(int(bright))

            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
