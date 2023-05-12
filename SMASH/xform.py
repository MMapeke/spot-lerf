import numpy as np


def normalize(a):
    """Normalizaes vector a.
    """
    if np.linalg.norm(a) == 0:
        return 0 * a
    return a / np.linalg.norm(a)


def get_world_to_local(local_frame):
    """Outputs xform matrix from lerf space to spot dock space
    """
    u = local_frame[0]
    xyz = local_frame[1:4]
    rot = np.zeros((4, 4), dtype=np.float32)
    rot[:3, :3] = xyz
    rot[3, 3] = 1.0
    tran = np.eye(4, dtype=np.float32)
    tran[:3, 3] = -u
    return np.matmul(rot, tran)


def construct_local_coord(A, B, C, D, orig):
    """Constructs local coordinate frame from x, y, z axes
    """
    xlate = orig - A
    A_xlated = A + xlate
    B_xlated = B + xlate
    C_xlated = C + xlate
    D_xlated = D + xlate

    x_axis_vec = normalize(A_xlated - D_xlated)
    y_axis_vec = normalize(B_xlated - A_xlated)
    z_axis_vec = normalize(A_xlated - C_xlated)

    local_coord = np.zeros((4, 3))
    local_coord[0] = orig
    local_coord[1] = x_axis_vec
    local_coord[2] = y_axis_vec
    local_coord[3] = z_axis_vec

    return local_coord


def get_scale(A, B, real_dist):
    """Computes scaling
    """
    nerf_dist = np.linalg.norm(A - B)
    return real_dist / nerf_dist


def transform(point, A, B, C, D, orig, real_dist):
    """Transforms a point from lerf space to spot dock space
    given reference points in lerf space
    """
    local_frame = construct_local_coord(A, B, C, D, orig)
    mat = get_world_to_local(local_frame)
    scale = get_scale(A, B, real_dist)
    h_point = np.ones((4, 1))
    h_point[:3, 0] = point
    xformed = scale * np.matmul(mat, h_point)
    return np.transpose(xformed)[0, :3]
