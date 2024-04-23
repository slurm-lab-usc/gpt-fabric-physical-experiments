import numpy as np


def solve_rigid_transform(camera_points_3d, robot_points_3d, debug=True):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    This is a (3, 4) matrix so we'd apply it on the original points in their homogeneous form with the fourth coordinate
    equal to one.

    Notation: A for camera points, B for robot points, so want to find an affine mapping from A -> B with orthogonal
    rotation and a translation.
    """

    assert camera_points_3d.shape[1] == robot_points_3d.shape[1] == 3
    A = camera_points_3d.T  # (3, N)
    B = robot_points_3d.T  # (3, N)

    # Look for Inge Soderkvist's solution online if confused
    meanA = np.mean(A, axis=1, keepdims=True)
    meanB = np.mean(B, axis=1, keepdims=True)
    A = A - meanA
    B = B - meanB
    covariance = B.dot(A.T)
    U, sigma, VH = np.linalg.svd(covariance)  # VH = V.T, i.e. numpy transposes it for us

    V = VH.T
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U.dot(V.T))
    R = U.dot(D).dot(V.T)
    t = meanB - R.dot(meanA)
    RB_matrix = np.concatenate((R, t), axis=1)

    if debug:  # sanity checks
        print("\nBegin debug prints for rigid transformation:")
        print("meanA:\n{}\nmeanB:\n{}".format(meanA, meanB))
        print("Rotation R:\n{}\nand R^TR (should be identity):\n{}".format(R, (R.T).dot(R)))
        print("translation t:\n{}".format(t))
        print("RB_matrix:\n{}".format(RB_matrix))

        # Get residual to inspect quality of solution. Use homogeneous coordinates for A.
        # Also, recall that we're dealing with (3,N) matrices, not (N,3).
        # In addition, we don't want to zero-mean for real applications.
        A = camera_points_3d.T  # (3,N)
        B = robot_points_3d.T  # (3,N)

        ones_vec = np.ones((1, A.shape[1]))
        A_h = np.concatenate((A, ones_vec), axis=0)
        B_pred = RB_matrix.dot(A_h)
        assert B_pred.shape == B.shape

        # Careful! Use raw_errors for the RF, but it will depend on pred-targ or targ-pred.
        raw_errors = B_pred - B  # Use pred-targ, of shape (3,N)

        print("\nCamera points (input), A.T:\n{}".format(A.T))
        print("Robot points (target), B.T:\n{}".format(B.T))
        print("Predicted robot points:\n{}".format(B_pred.T))
        print("Raw errors, B-B_pred:\n{}".format((B - B_pred).T))

    return RB_matrix


def convert_camera_points(camera_point, RB_matrix):
    camera_point = np.array([list(camera_point)])

    A = camera_point.T

    ones_vec = np.ones((1, A.shape[1]))
    A_h = np.concatenate((A, ones_vec), axis=0)
    B_pred = RB_matrix.dot(A_h)

    return B_pred
