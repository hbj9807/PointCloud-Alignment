import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.cm as cm

def compute_manual_transformation(source_points, target_points, with_scaling=True):
    """
    Computes the similarity transform (scale, rotation, translation)
    that best aligns two sets of corresponding points using the Umeyama method.
    
    Parameters:
        source_points: (N, 3) NumPy array of source points.
        target_points: (N, 3) NumPy array of target points.
        with_scaling:  If True, compute a scale factor; otherwise, use scale = 1.
    
    Returns:
        T: 4x4 transformation matrix.
        scale: The computed scale factor.
    """
    N = source_points.shape[0]
    
    # Compute centroids.
    mu_source = np.mean(source_points, axis=0)
    mu_target = np.mean(target_points, axis=0)
    
    # Center the points.
    X = source_points - mu_source
    Y = target_points - mu_target
    
    # Compute covariance using average (1/N) normalization.
    covariance = (Y.T @ X) / N
    
    # SVD decomposition.
    U, D, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R_mat = U @ S @ Vt
    
    # Compute scale if requested.
    if with_scaling:
        # Compute the average squared norm (variance) of the centered source points.
        sigma_X2 = np.sum(X**2) / N
        # Optimal scale: trace(D S) divided by the average squared norm.
        scale = np.sum(D * np.diag(S)) / sigma_X2
    else:
        scale = 1.0
    
    t = mu_target - scale * R_mat @ mu_source
    
    # Build the homogeneous transformation matrix.
    T = np.eye(4)
    T[:3, :3] = scale * R_mat
    T[:3, 3] = t
    return T, scale

def apply_depth_colormap(pcd, colormap_name='viridis'):
    """
    Applies a colormap to the point cloud based on the depth (z-coordinate) of each point.
    
    Parameters:
        pcd: An Open3D point cloud.
        colormap_name: Name of the matplotlib colormap to use (e.g., 'viridis', 'plasma').
    
    Returns:
        None. The pcd.colors attribute is updated.
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return
    # Use the z-coordinate as depth.
    z_vals = pts[:, 2]
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    print(f"Applying colormap: depth min = {z_min:.3f}, max = {z_max:.3f}")
    # Avoid division by zero.
    if abs(z_max - z_min) < 1e-8:
        norm = np.zeros_like(z_vals)
    else:
        norm = (z_vals - z_min) / (z_max - z_min)
    # Use the recommended API (for matplotlib >= 3.7)
    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    colors = cmap(norm)[:, :3]  # Use only RGB (ignore alpha)
    pcd.colors = o3d.utility.Vector3dVector(colors)

def pick_points(pcd, window_name, point_size=10.0):
    """
    Opens an interactive window to pick points from the point cloud using VisualizerWithEditing.
    
    Parameters:
        pcd: The Open3D point cloud.
        window_name: The title of the picking window.
        point_size: The size of the displayed points (increased for better visibility).
    
    Returns:
        picked_points: A NumPy array of the 3D coordinates of the selected points.
                       Returns None if fewer than 3 points were picked.
    """
    print(f"\n*** {window_name} ***")
    print("Please pick at least 3 corresponding points using [Shift + Left Click].")
    print("Then press 'q' to close the window.\n")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()  # Wait for user to pick points and press 'q'.
    vis.destroy_window()
    picked_ids = vis.get_picked_points()
    print("Picked point indices:", picked_ids)
    
    if len(picked_ids) < 3:
        print("Error: Not enough points were picked. You must pick at least 3 points.")
        return None
    
    points = np.asarray(pcd.points)
    picked_points = points[picked_ids, :]
    return picked_points

def main():
    # ------------------------------
    # Step 1. Load the point clouds.
    # ------------------------------
    source_path = r"Your .ply for the source ply file. "  # Unity
    target_path = r"Your .ply for the target ply file. "      # ROS Point Cloud
    
    source_pcd = o3d.io.read_point_cloud(source_path)
    target_pcd = o3d.io.read_point_cloud(target_path)
    
    # Apply a depth colormap to the point clouds so that each point gets a color based on its depth.
    apply_depth_colormap(source_pcd, colormap_name='viridis')
    apply_depth_colormap(target_pcd, colormap_name='plasma')
    
    # ------------------------------
    # Step 2. Manually pick corresponding points.
    # ------------------------------
    source_points = pick_points(source_pcd, "Source Point Cloud", point_size=3.0)
    if source_points is None:
        return
    
    target_points = pick_points(target_pcd, "Target Point Cloud", point_size=1.0)
    if target_points is None:
        return
    
    if source_points.shape[0] != target_points.shape[0]:
        print("Error: The number of picked points does not match between the source and target clouds.")
        return

    # Designate the very first selected point as the reference point.
    ref_source = source_points[0]
    ref_target = target_points[0]
    
    print("Reference Point (ROS's Position):", ref_target)
    
    # ------------------------------
    # Step 3. Compute the manual transformation.
    # ------------------------------
    T_manual, scale_factor = compute_manual_transformation(source_points, target_points, with_scaling=True)
    print("\nComputed Manual Transformation:")
    print(T_manual)
    print("\nComputed Scale Factor:", scale_factor)
    
    R_extracted = T_manual[:3, :3] / scale_factor
    euler_angles = R.from_matrix(R_extracted).as_euler('xyz', degrees=True)
    print("\nEuler Rotation (roll, pitch, yaw in degrees):", euler_angles)
    
    # ------------------------------
    # (Optional) Step 3.1: Refine using ICP.
    # ------------------------------
    threshold = 0.05  # Adjust this distance threshold as needed.
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, T_manual,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T_manual = reg_p2p.transformation
    print("\nRefined Transformation after ICP:")
    print(T_manual)
    
    # ------------------------------
    # Step 4. Apply the transformation to the source cloud.
    # ------------------------------
    source_pcd.transform(T_manual)
    
    # Re-apply the colormap after transformation in case depth distribution has changed.
    apply_depth_colormap(source_pcd, colormap_name='viridis')
    apply_depth_colormap(target_pcd, colormap_name='plasma')
    
    # ------------------------------
    # Step 5. Visualize the aligned point clouds using a custom Visualizer with a white background.
    # ------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Aligned Point Clouds", width=800, height=600)
    vis.add_geometry(source_pcd)
    vis.add_geometry(target_pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # White background for better contrast.
    # Force the visualizer to use the vertex colors.
    if hasattr(opt, 'point_color_option'):
        opt.point_color_option = o3d.visualization.PointColorOption.Color
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
