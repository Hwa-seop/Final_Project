import cv2
import numpy as np

cv2.namedWindow("Test Window")
cv2.imshow("Test Window", cv2.imread("test.jpg") or 255*np.ones((200, 200, 3), dtype=np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
