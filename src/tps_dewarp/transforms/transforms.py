import cv2
import numpy as np


# ===============================
# GLOBAL TRANSFORM
# ===============================

def apply_global_transform(img, rng):
    H, W = img.shape[:2]

    # rotation
    angle = rng.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_CUBIC,
        borderValue=255
    )

    # perspective
    margin = 0.08
    src = np.float32([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])

    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.uniform(-margin, margin) * W
        dst[i, 1] += rng.uniform(-margin, margin) * H

    H_mat = cv2.getPerspectiveTransform(src, dst)

    img = cv2.warpPerspective(
        img, H_mat, (W, H),
        flags=cv2.INTER_CUBIC,
        borderValue=255
    )

    return img


# ===============================
# TPS IMPLEMENTATION
# ===============================

def U_func(r2):
    r2[r2 == 0] = 1e-6
    return r2 * np.log(r2)


def compute_tps_coefficients(src_pts, dst_pts):
    n = src_pts.shape[0]

    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            r2 = np.sum((src_pts[i] - src_pts[j])**2)
            K[i, j] = U_func(np.array([r2]))[0]

    P = np.hstack((np.ones((n, 1)), src_pts))
    O = np.zeros((3, 3))

    L = np.vstack((
        np.hstack((K, P)),
        np.hstack((P.T, O))
    ))

    Y = np.vstack((dst_pts, np.zeros((3, 2))))

    params = np.linalg.solve(L, Y)
    return params


def tps_warp_image(img, src_pts, dst_pts):
    H, W = img.shape[:2]
    params = compute_tps_coefficients(src_pts, dst_pts)

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    n = src_pts.shape[0]

    U = np.zeros((points.shape[0], n))
    for i in range(n):
        r2 = np.sum((points - src_pts[i])**2, axis=1)
        U[:, i] = U_func(r2)

    P = np.hstack((np.ones((points.shape[0], 1)), points))

    mapped = U @ params[:n] + P @ params[n:]
    map_x = mapped[:, 0].reshape(H, W).astype(np.float32)
    map_y = mapped[:, 1].reshape(H, W).astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    return warped


# ===============================
# TPS WAVE (5x5)
# ===============================

def apply_tps_wave(img, rng):
    H, W = img.shape[:2]

    gx = np.linspace(0, W - 1, 5)
    gy = np.linspace(0, H - 1, 5)
    src_pts = np.array([(x, y) for y in gy for x in gx], np.float64)

    dst_pts = src_pts.copy()

    amp_x = rng.uniform(5, 20)
    amp_y = rng.uniform(5, 15)
    freq_x = rng.uniform(1.0, 2.0)
    freq_y = rng.uniform(1.0, 2.0)

    for i, (x, y) in enumerate(src_pts):
        dx = amp_x * np.sin(2 * np.pi * y / H * freq_y)
        dy = amp_y * np.sin(2 * np.pi * x / W * freq_x)
        dst_pts[i] += [dx, dy]

    warped = tps_warp_image(img, dst_pts, src_pts)  # обратное отображение
    return warped


# ===============================
# MAIN PIPELINE
# ===============================

def generate_warped_image(input_path, output_path, random_seed=None):
    rng = np.random.default_rng(random_seed)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    img = apply_global_transform(img, rng)
    img = apply_tps_wave(img, rng)

    cv2.imwrite(output_path, img)

    return output_path
