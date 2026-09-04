import cv2
import numpy as np
import random

IMAGE_PATH = "images/1.png"
DISPLAY_WIDTH = 1000
POINTS_PER_FRAME = 30

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: image not found. Check your file path.")
    exit()

height, width, channels = image.shape

kernel = np.ones((3, 3), np.uint8)

# Edge detection at full original resolution
b, g, r = cv2.split(image)
edges_b = cv2.Canny(cv2.GaussianBlur(b, (3, 3), 0), 20, 60)
edges_g = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 20, 60)
edges_r = cv2.Canny(cv2.GaussianBlur(r, (3, 3), 0), 20, 60)
edges = cv2.bitwise_or(edges_b, cv2.bitwise_or(edges_g, edges_r))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

colored_edges = np.zeros_like(image)
colored_edges[edges > 0] = image[edges > 0]

# Find contours at full resolution
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Only resize for display — quality stays the same regardless of DISPLAY_WIDTH
scale = DISPLAY_WIDTH / width
display_height = int(height * scale)

colored_edges_small = cv2.resize(colored_edges, (DISPLAY_WIDTH, display_height))
edges_small = cv2.resize(edges, (DISPLAY_WIDTH, display_height))

# Scale contour points to display size
contour_points = []
for contour in contours:
    pts = [(int(pt[0][1] * scale), int(pt[0][0] * scale)) for pt in contour]
    contour_points.append(pts)

random.shuffle(contour_points)

small_height, small_width = edges_small.shape
canvas = np.zeros((small_height, small_width, 3), dtype=np.uint8)

cv2.namedWindow("Drawing", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Drawing", canvas)
cv2.waitKey(1)

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
