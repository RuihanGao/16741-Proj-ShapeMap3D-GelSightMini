import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os.path as osp
import scipy
import open3d as o3d
from scipy.spatial.distance import cdist
import scipy.io

"""
Python implementation that is equivalent to sampleModelPoints function in generateHeightMap.m
Author: Ruihan Gao
Date: Nov, 2023

Change log:
2023.12.01 since the size of the calibration ball is too small, the scale and sampling of l and b implemented in the original script is not suitable for the calibration ball. Thus new parameters such as z_margin and bl_margin are added to cater for different scenarios. When calib_ball is the object, we set the margin based on the ball radius. Otherwise, we follow the original setting for existing objects in the dataset.
2023.12.01 Avoid RunTime Warning in point_to_line_distance by avoiding division by zero
2023.12.01 Add hard_code flag to hard code the safe_pts for calibration ball for debugging (sample one point at the top of the ball)
"""


### helper functions ###
def rotate_point_cloud_along_z(pc, direction):
    """
    Rotate the point cloud along z-axis to align the highest eigenvector with the z-axis. 
    For the 2nd highest eigenvector, the user can choose to align it with the x-axis or y-axis.

    Args:
        pc: point cloud, N * 3
        direction: 'x' or 'y' (align the 2nd highest eigenvector with x-axis or y-axis)

    Returns:
        pc2: rotated point cloud, N * 3
    """
    # Bring the point cloud center to the origin
    pc = pc - np.mean(pc, axis=0)

    # Calculate the eigenvector of the highest eigenvalue
    u = pca_eig(pc, 'max')

    # Calculate the angles of the normal vector
    alpha, beta = unit_vector_to_angle(u)

    # Align the point cloud along x-axis followed by aligning along z-axis
    _, Ry, Rz = rotational_matrix(-alpha, np.pi - beta)
    pc2 = rotate_pc(pc, Ry, Rz)

    # Align v normal vector along x or y direction
    offset = 0 if direction == 'x' else np.pi / 2
    v = pca_eig(pc2, 'middle')

    # Calculate the angle of the projected v-vector along the xy-plane
    alpha, _ = unit_vector_to_angle(v)

    # Calculate the rotational matrix for the angle
    _, Ry, Rz = rotational_matrix(offset - alpha, 0)

    # Rotate the point cloud
    pc2 = rotate_pc(pc2, Ry, Rz)
    return pc2

def rotational_matrix(alpha, beta):
    Rx = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    return Rx, Ry, Rz

def pca_eig(pc, magnitude):
    # Obtain the covariance matrix
    covariance = np.cov(pc, rowvar=False)

    # Calculate the eigenvectors and obtain the normal vector
    _, V = np.linalg.eig(covariance)

    # Sort the eigenvectors based on size of eigenvalues
    idx = np.argsort(np.abs(np.linalg.eigvals(covariance)))[::-1]
    V = V[:, idx]

    if magnitude == 'max':
        return V[:, 2]
    elif magnitude == 'middle':
        return V[:, 1]
    elif magnitude == 'min':
        return V[:, 0]

def unit_vector_to_angle(u):
    # Rotational angle between the projected u on the xy plane and the x-axis
    alpha = np.arctan2(u[1], u[0])

    # Rotational angle between the u vector and the z-axis
    beta = np.arctan2(np.sqrt(u[0]**2 + u[1]**2), u[2])

    return alpha, beta

def rotate_pc(pc, Ry, Rz):
    """
    Rotate the point cloud.
    Args:
        pc: point cloud, N * 3
        Ry: rotational matrix along y-axis, shape [3, 3]
        Rz: rotational matrix along z-axis, shape [3, 3]
    """
    # Convert the point cloud to 3 * N format
    matrix = pc.T

    # Apply the rotation matrix
    matrix2 = np.dot(Ry, np.dot(Rz, matrix))

    # Output the point cloud
    pc2 = matrix2.T

    return pc2

def point_to_line_distance(points, origin, target):
    # Calculate the distance from points to a line defined by origin and target
    # The points array should have shape (N, 3)
    # The origin and target should be 1D arrays with shape (3,)
    direction = target - origin
    magnitude = np.linalg.norm(direction)
    if magnitude == 0:
        return np.linalg.norm(points - origin, axis=1)
    direction /= magnitude
    distances = np.linalg.norm(np.cross(points - origin, direction), axis=1)
    return distances

def find_closest_point_on_line(points, origin, target):
    distances = point_to_line_distance(points, origin, target)
    return np.argmin(distances)

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

### helper functions end ###




shape_mapping_root_path = osp.normpath(osp.join(osp.abspath(__file__), '..', '..', '..', '..'))
print(f"shape_mapping_root_path: {shape_mapping_root_path}")

def sample_model_points(obj, save_flag, verbose=False, ball_radius=None, hard_code=False):
    key_set = ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
               "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
               "010_potted_meat_can", "011_banana", "012_strawberry", "013_apple", "014_lemon",
               "017_orange", "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "025_mug",
               "029_plate", "035_power_drill", "036_wood_block", "037_scissors", "042_adjustable_wrench",
               "043_phillips_screwdriver", "048_hammer", "055_baseball", "056_tennis_ball",
               "072-a_toy_airplane", "072-b_toy_airplane", "077_rubiks_cube", "calib_ball"]
    value_set = ["top_circle", "top_line", "top_line", "top_circle",
                 "top_circle", "top_circle", "top_line", "top_line",
                 "top_line", "top_line", "top_line", "top_line", "top_line",
                 "top_line", "top_circle", "top_circle", "top_line", "top_line",
                 "top_line", "top_circle", "top_circle", "top_line", "top_line",
                 "top_line", "top_line", "top_line", "top_line",
                 "top_line", "top_line", "top_line", "top_circle"]
    top_map = dict(zip(key_set, value_set))

    # N heights, M angles, K top/bottom = 5*8 + 5 + 5 = 50 contacts
    if obj in ["021_bleach_cleanser", "036_wood_block"]:
        N = 6
        M = 8
    else:
        N = 5
        M = 8   
    num_samples = 60 # either 6*8+12 or 5*8+20


    # Load .mat object file in python
    data_path = os.path.join(shape_mapping_root_path, 'models', obj, 'textured.mat')
    data_dict = scipy.io.loadmat(data_path) # ['__header__', '__version__', '__globals__', 'None', 'vertices', 'normals', 'faces', '__function_workspace__']

    XObj = data_dict['vertices'] # [N,3] # [262144, 3] for calib_ball
    YObj = data_dict['normals'] # [N,3]
    TriObj = data_dict['faces']
    
    count = len(XObj) # N
      # Subsample dense point cloud
    # ceate a PointCloud object from the list of points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(XObj)
    point_cloud.normals = o3d.utility.Vector3dVector(YObj)
    if verbose:
        # visualize the point cloud (optional)
        print(f"Visualize the original point cloud ...")
        o3d.visualization.draw_geometries([point_cloud])

    # get min and max bound of the point cloud
    x_min, y_min, z_min = point_cloud.get_min_bound()
    x_max, y_max, z_max = point_cloud.get_max_bound()

    m_limit = np.mean([x_max - x_min, y_max - y_min, z_max - z_min])
    # Downsample the point cloud
    scale_factor = 1
    new_cloud = o3d.geometry.PointCloud()
    while len(new_cloud.points) < num_samples:
        new_cloud = point_cloud.voxel_down_sample(voxel_size=m_limit / (scale_factor * 100))
        scale_factor += 1
    if verbose: print(f"check point_cloud size: {len(point_cloud.points)}, downsampled point cloud {len(new_cloud.points)}, scale_factor: {scale_factor}")
    point_cloud = new_cloud
    if verbose:
        # visualize the downsampled point cloud
        print(f"Visualize the downsampled point cloud ...")
        o3d.visualization.draw_geometries([point_cloud])


    # Sample contact points from the side and top, as described in the paper
    points = np.asarray(point_cloud.points)
    
    # Rotate the point cloud to align the highest eigenvector with the z-axis
    if obj == "010_potted_meat_can" :
        print(f"Don't rotate the point cloud")
        r_cloud = np.asarray(points)
        rotate_dir = 'z'
    elif obj == "004_sugar_box":
        print(f"direction x")
        r_cloud = rotate_point_cloud_along_z(points, 'x')  # Rotate cloud along positive Z
        rotate_dir = 'x'
    else:
        print(f"direction y")
        r_cloud = rotate_point_cloud_along_z(points, 'y')  # Rotate cloud along positive Z
        rotate_dir = 'y'
    
    # Sample the contact points
    if not hard_code:
        safe_pts = []
        # Set the parameters for points sampling
        radius = 0.2 if ball_radius is None else ball_radius
        z_margin = ball_radius / 10.0 if ball_radius is not None else 1e-2

        if ball_radius is not None:
            bl_margin = ball_radius/20.0
        elif obj == "004_sugar_box":
            bl_margin = 1e-2
        elif obj == "010_potted_meat_can" or obj == "005_tomato_soup_can":
            bl_margin = 1.5e-2
        else:
            bl_margin = 2e-2
        
        l = (np.max(r_cloud[:, 0]) - np.min(r_cloud[:, 0])) / 2 - bl_margin
        b = (np.max(r_cloud[:, 1]) - np.min(r_cloud[:, 1])) / 2 - bl_margin

        z_heights = np.linspace(np.min(r_cloud[:, 2]) + z_margin, np.max(r_cloud[:, 2]) - z_margin, N)
        
        print(f"l: {l}, b: {b}")
        print(f"z_heights: {z_heights}")

        # Get side points = N *M points
        for z in z_heights:
            if obj == "004_sugar_box":
                angles = np.deg2rad([0, 30, 90, 150, 180, 210, 270, 330])
            elif obj == "010_potted_meat_can":
                angles = np.deg2rad([0, 45, 90, 135, 180, 215, 270, 315])
            else:
                angles = np.linspace(0, 2 * np.pi, M + 1)
                angles = angles[:M]

            x_center = np.mean(r_cloud[:, 0])
            y_center = np.mean(r_cloud[:, 1])
            x = radius * np.cos(angles) + x_center
            y = radius * np.sin(angles) + y_center
            new_pts = np.column_stack((x, y, np.ones_like(x) * z))
            safe_pts.append(new_pts)
        safe_pts = np.vstack(safe_pts) # [M*N, 3] = [40, 3]

        c = np.cos(np.deg2rad(45))

        
        # Get top points
        if top_map[obj] == 'top_line':
            if obj == "010_potted_meat_can":
                x_vals = np.linspace(np.min(r_cloud[:, 0]) + 1.5e-2, np.max(r_cloud[:, 0]) - 1.5e-2, 5)
                y_vals = np.linspace(np.min(r_cloud[:, 1]) + 1.5e-2, np.max(r_cloud[:, 1]) - 1.5e-2, 2)
            elif obj == "004_sugar_box":
                x_vals = np.linspace(np.min(r_cloud[:, 0]) + 1e-2, np.max(r_cloud[:, 0]) - 1e-2, 5)
                y_vals = np.linspace(np.min(r_cloud[:, 1]) + 1e-2, np.max(r_cloud[:, 1]) - 1e-2, 2)
            else:
                x_vals = np.linspace(np.min(r_cloud[:, 0]), np.max(r_cloud[:, 0]), 5 + 2)[1:5 + 1]
                y_vals = np.linspace(np.min(r_cloud[:, 1]), np.max(r_cloud[:, 1]), 2 + 2)[1:2 + 1]
            
            top_pts = np.array([
                [x_vals[0], y_vals[0], radius],
                [x_vals[1], y_vals[0], radius],
                [x_vals[2], y_vals[0], radius],
                [x_vals[3], y_vals[0], radius],
                [x_vals[4], y_vals[0], radius],
                [x_vals[0], y_vals[1], radius],
                [x_vals[1], y_vals[1], radius],
                [x_vals[2], y_vals[1], radius],
                [x_vals[3], y_vals[1], radius],
                [x_vals[4], y_vals[1], radius],
                [x_vals[0], y_vals[0], -radius],
                [x_vals[1], y_vals[0], -radius],
                [x_vals[2], y_vals[0], -radius],
                [x_vals[3], y_vals[0], -radius],
                [x_vals[4], y_vals[0], -radius],
                [x_vals[0], y_vals[1], -radius],
                [x_vals[1], y_vals[1], -radius],
                [x_vals[2], y_vals[1], -radius],
                [x_vals[3], y_vals[1], -radius],
                [x_vals[4], y_vals[1], -radius]
            ])
        elif obj in ["021_bleach_cleanser", "036_wood_block"]:
            top_pts = np.array([
                [np.mean(r_cloud[:, 0]) + 1e-2, np.mean(r_cloud[:, 1]), radius],
                [np.mean(r_cloud[:, 0]) + 1e-2, np.mean(r_cloud[:, 1]), -radius],
                [np.mean(r_cloud[:, 0]) - 1e-2, np.mean(r_cloud[:, 1]), radius],
                [np.mean(r_cloud[:, 0]) - 1e-2, np.mean(r_cloud[:, 1]), -radius],
                [c * l, c * b, radius],
                [c * l, -c * b, radius],
                [-c * l, -c * b, radius],
                [-c * l, c * b, radius],
                [c * l, c * b, -radius],
                [c * l, -c * b, -radius],
                [-c * l, -c * b, -radius],
                [-c * l, c * b, -radius]
            ])
        else:
            top_pts = np.array([
                [np.mean(r_cloud[:, 0]) + 1e-2, np.mean(r_cloud[:, 1]), radius],
                [np.mean(r_cloud[:, 0]) + 1e-2, np.mean(r_cloud[:, 1]), -radius],
                [np.mean(r_cloud[:, 0]) - 1e-2, np.mean(r_cloud[:, 1]), radius],
                [np.mean(r_cloud[:, 0]) - 1e-2, np.mean(r_cloud[:, 1]), -radius],
                [c * l, c * b, radius],
                [c * l, -c * b, radius],
                [-c * l, -c * b, radius],
                [-c * l, c * b, radius],
                [c * l, c * b, -radius],
                [c * l, -c * b, -radius],
                [-c * l, -c * b, -radius],
                [-c * l, c * b, -radius],
                [l, np.mean(r_cloud[:, 1]), radius],
                [-l, np.mean(r_cloud[:, 1]), radius],
                [np.mean(r_cloud[:, 0]), b, radius],
                [np.mean(r_cloud[:, 0]), -b, radius],
                [l, np.mean(r_cloud[:, 1]), -radius],
                [-l, np.mean(r_cloud[:, 1]), -radius],
                [np.mean(r_cloud[:, 0]), b, -radius],
                [np.mean(r_cloud[:, 0]), -b, -radius]
            ])

        safe_pts = np.vstack((safe_pts, top_pts))


        # # Get top/bottom points = K + K
        # # TODO: Note by Ruihan: the org implementation comments out the top/bottom sampling
        # K = 5
        # isTopBottom = False 
        
        # # get the 90 percentile and 10 percentile of the point cloud height as the top and bottom line
        # top = np.percentile(r_cloud[:, 2], 90)
        # bot = np.percentile(r_cloud[:, 2], 10)
        # top_id = np.where(r_cloud[:, 2] > top)[0]
        # bot_id = np.where(r_cloud[:, 2] < bot)[0]
        
        # if isTopBottom:
        #     while True:
        #         # check for angles
        #         samp_id = np.random.choice(top_id, K, replace=False)
        #         samp_n = point_cloud.Normal[samp_id, :]
        #         flag = True
        #         for i in range(K):
        #             ni = samp_n[i, :]
        #             for j in range(K):
        #                 nj = samp_n[j, :]
        #                 ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(ni, nj)), np.dot(ni, nj)))
        #                 if ang > 60:
        #                     flag = False
        #                     break
        #             if not flag:
        #                 break
        #         if flag:
        #             sampleTopID = samp_id
        #             break

        #     while True:
        #         samp_id = np.random.choice(bot_id, K, replace=False)
        #         samp_n = point_cloud.Normal[samp_id, :]
        #         flag = True
        #         for i in range(K):
        #             ni = samp_n[i, :]
        #             for j in range(K):
        #                 nj = samp_n[j, :]
        #                 ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(ni, nj)), np.dot(ni, nj)))
        #                 if ang > 60:
        #                     flag = False
        #                     break
        #             if not flag:
        #                 break
        #         if flag:
        #             sampleBotID = samp_id
        #             break
        # else:
        #     skip = round(len(top_id) / K)
        #     rIdx = np.arange(0, len(top_id), skip)
        #     sampleTopID = top_id[rIdx]

        #     skip = round(len(bot_id) / K)
        #     rIdx = np.arange(0, len(bot_id), skip)
        #     sampleBotID = bot_id[rIdx]

        # top_pts = r_cloud[sampleTopID, :]
        # bot_pts = r_cloud[sampleBotID, :]

    else:
        # hard code the safe_pts for calibration ball for debugging
        safe_pts = []
        assert ball_radius is not None, "ball_radius should not be None"
        # find the top point of the sphere
        top_pt = np.array([np.mean(r_cloud[:, 0]), np.mean(r_cloud[:, 1]), np.max(r_cloud[:, 2])])
        safe_pts.append(top_pt)
        safe_pts = np.array(safe_pts)
    if verbose:
        print(f"Find {len(safe_pts)} safe points: {safe_pts}")




    ### Given the safe_pts, find the id of the closest point on the point cloud
    sample_side_id = []
    for i in range(safe_pts.shape[0]):
        # get closest point on the vector
        if i <= M * N:
            # side points
            origin = np.array([np.mean(r_cloud[:, 0]), np.mean(r_cloud[:, 1]), safe_pts[i, 2]])
        else:
            # top points
            origin = np.array([safe_pts[i, 0], safe_pts[i, 1], np.mean(r_cloud[:, 2])])

        dist = point_to_line_distance(r_cloud, origin, safe_pts[i, :])
        pos = np.argsort(dist)[:100]
        p = pos[0]

        for j in range(len(pos)):
            if np.linalg.norm(safe_pts[i, :] - r_cloud[pos[j], :]) < np.linalg.norm(safe_pts[i, :] - origin):
                p = pos[j]
                break
        sample_side_id.append(p)


    # Given the sample_id, retrieve the points and normals from point_cloud

    # Access normals as a NumPy array (if available)
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
    else:
        normals = None

    sample_points = points[sample_side_id, :]
    assert normals is not None, "sampled normals should not be None"
    sample_normals = normals[sample_side_id, :]

    # sort the points and return the idx
    indices = np.argsort(sample_points[:,2])
    sample_points = sample_points[indices]
    sample_normals = sample_normals[indices]
    if verbose:
        print(f"check example of sample_points {sample_points[0]}, sample_normals {sample_normals[0]}")


    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Plot the sampled points and their normals in 3D plot
    print("Plot the sampled points and their normals in 3D plot ...")
    quiver_length = 0.5 * ball_radius if ball_radius is not None else 0.02
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], s=1, c='b')
    ax.quiver(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
              sample_normals[:, 0], sample_normals[:, 1], sample_normals[:, 2], length=quiver_length, color='b', alpha=0.5)

    # Plot the triangular surface
    # subtract 1 from all point indices in TriObj, except for calib_ball
    # When Ruihan generated the faces for calib_bal, the starting index is 0
    if obj != "calib_ball":
        TriObj_ = TriObj - 1
    else:
        TriObj_ = TriObj
    ax.plot_trisurf(XObj[:, 0], XObj[:, 1], XObj[:, 2], triangles=TriObj_, shade=True, color='gray', alpha=0.3, edgecolor='none')

    set_axes_equal(ax) 
    # rotate the plot based on rotate_dir so that we have z-axis pointing up
    if rotate_dir == 'x':
        ax.view_init(0, 0)
    elif rotate_dir == 'y':
        ax.view_init(0, 90)
    else:
        ax.view_init(90, 0)
    
    # rotate the view again so that the camera is looking down 45 degrees facing the object
    ax.view_init(elev=45)
    # rotate the camera 45 to the side 
    ax.view_init(azim=45)
    ax.axis('off')

    if save_flag:
        # Save the plot and sampled data
        num_samples = len(sample_points)
        print(f"Saving sampled points to .mat file, find {num_samples} points")
        model_path = os.path.join(shape_mapping_root_path, 'models', obj)
        model_name = data_path.split('/')[-1].split('.')[0]

        obj_name = '_'.join(obj.split('_')[1:]) # get rid of the number prefix
        dataset_path = os.path.join(shape_mapping_root_path, 'data', obj_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Save output to .mat file to be consistent as original implementation
        scipy.io.savemat(os.path.join(model_path, f'{model_name}_{num_samples}sampled_python.mat'),
                         {'model_name': model_name,
                          'vertices': XObj,
                          'normals': YObj,
                          'faces': TriObj,
                          'numSamples': num_samples,
                          'samplePoints': sample_points,
                          'sampleNormals': sample_normals})
        # Save another copy to data folder
        scipy.io.savemat(os.path.join(dataset_path, f'{obj_name}_{num_samples}sampled_python.mat'),
                         {'model_name': model_name,
                          'vertices': XObj,
                          'normals': YObj,
                          'faces': TriObj,
                          'numSamples': num_samples,
                          'samplePoints': sample_points,
                          'sampleNormals': sample_normals})        
        # Note by RH: add the postfix _python to the saved file so that we can compare with the original implementation
        print(f"Saving sampled points {os.path.join(model_path, f'{model_name}_{num_samples}sampled_python.png')}")
        plt.savefig(os.path.join(model_path, f'{model_name}_{num_samples}sampled_python.png'), dpi=300)
        print(f"Saving sampled points {os.path.join(dataset_path, f'{obj_name}_{num_samples}sampled_python.png')}")
        plt.savefig(os.path.join(dataset_path, f'{obj_name}_{num_samples}sampled_python.png'), dpi=300) 
    else:
        plt.show()
    plt.close()



if __name__ == "__main__": 
    obj_name = "006_mustard_bottle"
    sample_model_points(obj_name, save_flag=True, verbose=False)

    # obj_name = "calib_ball"
    # sample_model_points(obj_name, save_flag=True, verbose=False, hard_code=False, ball_radius=0.003) #  # the ball radius should be consistent as tha value in create_ball_mesh.ipynb

