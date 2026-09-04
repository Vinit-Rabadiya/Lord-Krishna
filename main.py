import cv2
import numpy as np
import random

IMAGE_PATH = "images/birth.jpg"
DISPLAY_HEIGHT = 750
POINTS_PER_FRAME = 30

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: image not found. Check your file path.")
    exit()

height, width, channels = image.shape
scale = DISPLAY_HEIGHT / height
display_width = int(width * scale)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)

blurred = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blurred, 30, 100)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

colored_edges = np.zeros_like(image)
colored_edges[edges > 0] = image[edges > 0]

colored_edges_small = cv2.resize(colored_edges, (display_width, DISPLAY_HEIGHT))
edges_small = cv2.resize(edges, (display_width, DISPLAY_HEIGHT))

contours, _ = cv2.findContours(edges_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

small_height, small_width = edges_small.shape
canvas = np.zeros((small_height, small_width, 3), dtype=np.uint8)

# Convert to (y, x) point lists and shuffle so drawing order is random
contour_points = []
for contour in contours:
    pts = [(pt[0][1], pt[0][0]) for pt in contour]
    contour_points.append(pts)

random.shuffle(contour_points)

cv2.namedWindow("Drawing", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Drawing", canvas)
cv2.waitKey(1)

# Draw one contour at a time, POINTS_PER_FRAME points per screen refresh
for contour in contour_points:
    for i in range(0, len(contour), POINTS_PER_FRAME):
        batch = contour[i:i + POINTS_PER_FRAME]
        for y, x in batch:
            if 0 <= y < small_height and 0 <= x < small_width:
                canvas[y, x] = colored_edges_small[y, x]
        cv2.imshow("Drawing", canvas)
        cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()
