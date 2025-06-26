import cv2
import numpy as np
def dummy(event, x, y, flags, param):
    pass
cv2.namedWindow("test")
cv2.setMouseCallback("test", dummy)
while True:
    img = 255 * np.ones((200, 200, 3), dtype=np.uint8)
    cv2.imshow("test", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()