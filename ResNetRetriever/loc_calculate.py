import cv2
import numpy as np

def compute_optical_flow_Farneback(img1, img2):
    """Compute optical flow using Farneback method."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    return u, v

def visualize_optical_flow(u, v, step=16):
    """Visualize optical flow using quiver plot (arrows)."""
    h, w = u.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = int(x + u[y, x]), int(y + v[y, x])
            cv2.arrowedLine(output, (x, y), (fx, fy), (0, 255, 0), 1, tipLength=0.3)
    return output

# Example usage
img1 = cv2.imread("/home/yuhai/ResNetRetriever/class1/1_4.JPG")
img2 = cv2.imread("/home/yuhai/ResNetRetriever/class2/2_1.JPG")

u, v = compute_optical_flow_Farneback(img1, img2)
flow_img = visualize_optical_flow(u, v)

cv2.imshow("Optical Flow", flow_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

