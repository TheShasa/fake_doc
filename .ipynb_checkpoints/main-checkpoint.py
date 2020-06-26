import cv2
import numpy as np

name = '8af48251-dc69-414e-9f69-65b03eb09e59.jpg'
img = cv2.imread(name)

edges = cv2.Canny(img, 100, 250)
#
# cv2.imshow('edg',edges)
# cv2.waitKey(0)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 10)
print(lines)
for line in (lines):
    rho, theta = line[0][0], line[0][1]
    print(2)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    print(x1,y1,x2,y2)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('houghlines3.jpg', img)
