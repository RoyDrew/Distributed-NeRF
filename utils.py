from scipy.spatial.transform import Rotation
import numpy as np
import cv2

def create_alpha_fn(alpha0, warm_up, linear):
    def alpha_fn1(progress):
        return alpha0 + (1-alpha0) / (1 + np.exp(-32 * (progress - warm_up)))
    def alpha_fn2(progress):
        return min(alpha0+(1-alpha0+0.3)*progress, 1.0)
    return alpha_fn2 if linear else alpha_fn1

def rot_phi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]])

def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])

def rot_psi(psi): return np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

def trans_t(t): return np.array([
    [1, 0, 0, t[0]],
    [0, 1, 0, t[1]],
    [0, 0, 1, t[2]],
    [0, 0, 0, 1]])

def get_noised_pose(c2w, delta):
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    #w2c = np.linalg.inv(np.concatenate([c2w, bottom], axis=0))
    
    x, y, z, phi, theta, psi = delta
    t = (x, y, z)
    noised_pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi) @ trans_t(t) @ np.concatenate([c2w, bottom], axis=0)
    #noised_pose = np.linalg.inv(noised_pose)
    noised_pose = noised_pose[:3, :4]
    return noised_pose

def extract_delta(c2w) -> np.ndarray:
    np.array(c2w)
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    r = Rotation.from_matrix(R)
    euler_angles = r.as_euler('zyx', degrees=True)
    return euler_angles, T

def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected points", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

def find_Edge(img_rgb, DEBUG=False):
    # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用Canny边缘检测方法
    edges = cv2.Canny(img_gray, 150, 450)

    # 获取边缘点的坐标
    xy = np.argwhere(edges > 0)
    xy = np.array(xy).astype(int)

    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)

    return xy  # pixel coordinates

def find_EdgeRegion(img_rgb, DEBUG=False):
    # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用Canny边缘检测方法
    edges = cv2.Canny(img_gray, 150, 450)

    # 使用cv2.dilate()进行膨胀操作
    search_radius = 7
    kernel = np.ones((search_radius, search_radius), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    
    if DEBUG:
        cv2.imwrite("Detected edges", dilated_edges)

    # 获取膨胀后的边缘点的坐标
    xy = np.argwhere(dilated_edges > 0)
    xy = np.array(xy).astype(int)

    return xy  # pixel coordinates