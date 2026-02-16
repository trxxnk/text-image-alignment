import numpy as np
import cv2


def _U_func(r2):
    # r^2 * log(r^2), с защитой от log(0)
    r2 = np.maximum(r2, 1e-6)
    return r2 * np.log(r2)


def _compute_tps_params(src_pts, dst_pts):
    """
    src_pts, dst_pts: (K, 2) в пикселях
    возвращает параметры TPS
    """
    K = src_pts.shape[0]

    # K x K
    diff = src_pts[:, None, :] - src_pts[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K_mat = _U_func(r2)

    # K x 3
    P = np.concatenate(
        [np.ones((K, 1)), src_pts],
        axis=1
    )

    # (K+3) x (K+3)
    L = np.zeros((K + 3, K + 3), dtype=np.float64)
    L[:K, :K] = K_mat
    L[:K, K:] = P
    L[K:, :K] = P.T

    Y = np.zeros((K + 3, 2), dtype=np.float64)
    Y[:K] = dst_pts

    params = np.linalg.solve(L, Y)
    return params


def _apply_tps(points, src_pts, params):
    """
    points: (N, 2)
    src_pts: (K, 2)
    params: (K+3, 2)
    """
    K = src_pts.shape[0]

    diff = points[:, None, :] - src_pts[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    U = _U_func(r2)

    P = np.concatenate(
        [np.ones((points.shape[0], 1)), points],
        axis=1
    )

    warped = U @ params[:K] + P @ params[K:]
    return warped


def build_remap_from_delta_tps(
    delta_tps: np.ndarray,
    H: int,
    W: int,
    grid_size: int = 5,
    clip: bool = True
):
    """
    delta_tps: (25, 2), нормализованные смещения
    H, W: размеры изображения
    """

    assert delta_tps.shape == (grid_size * grid_size, 2)

    # --- базовая нормализованная сетка ---
    s = np.linspace(0.0, 1.0, grid_size)
    t = np.linspace(0.0, 1.0, grid_size)
    P_base_norm = np.array(
        [(sx, ty) for ty in t for sx in s],
        dtype=np.float32
    )

    # --- искривлённая сетка ---
    P_warp_norm = P_base_norm + delta_tps

    if clip:
        P_warp_norm = np.clip(P_warp_norm, 0.0, 1.0)

    # --- перевод в пиксели ---
    P_base_px = np.column_stack([
        P_base_norm[:, 0] * (W - 1),
        P_base_norm[:, 1] * (H - 1)
    ])

    P_warp_px = np.column_stack([
        P_warp_norm[:, 0] * (W - 1),
        P_warp_norm[:, 1] * (H - 1)
    ])

    # --- строим TPS: warp -> base ---
    params = _compute_tps_params(
        src_pts=P_warp_px,
        dst_pts=P_base_px
    )

    # --- строим dense remap ---
    grid_y, grid_x = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )
    grid = np.stack(
        [grid_x.ravel(), grid_y.ravel()],
        axis=1
    )

    warped = _apply_tps(grid, P_warp_px, params)

    map_x = warped[:, 0].reshape(H, W).astype(np.float32)
    map_y = warped[:, 1].reshape(H, W).astype(np.float32)

    return map_x, map_y
