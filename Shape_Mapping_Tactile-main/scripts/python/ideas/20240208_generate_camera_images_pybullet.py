import pybullet as p
import pybullet_data
import numpy as np

# Initialize PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # set the search path for data files

# Load object mesh
object_id = p.loadURDF("/home/ruihan/Documents/16741_proj/Shape_Mapping_Tactile-main/data/google_512k_012_strawberry/strawberry.urdf", basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])

# # Add visual texture to the object mesh
# texture_path = "/home/ruihan/Documents/16741_proj/Shape_Mapping_Tactile-main/data/mustard_bottle/texture_map.png.png"  # replace with the path to your texture file
# texture_id = p.loadTexture(texture_path)
# p.changeVisualShape(object_id, -1, textureUniqueId=texture_id)

# Define camera parameters
width = 640
height = 480
fov = 60
aspect = width / height
near = 0.001
far = 1.0
camera_pos = [0.1, 0.1, 0.1]  # camera position
target_pos = [0, 0, 0]  # where the camera is pointing towards
up_vector = [0, 0, 1]  # up direction of the camera

# Main loop
while True:
    # Set up the camera
    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
    print(f"check view_matrix: {type(view_matrix)} \n{view_matrix}")

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    print(f"check projection_matrix: {projection_matrix.shape} \n{projection_matrix}")

    # Note: if you want to convert this projection matrix to a camera intrinsics and extrinsics, reference here: https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
    # I haven't tried, switching to pyrender and trimesh since we are not considering and force interaction with the object.


    # Render the scene
    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    print(f"check images: \n{np.array(images).shape}")    

    # Convert the image data to a numpy array
    img_arr = np.array(images[2])
    # save the image
    import cv2
    cv2.imwrite("image.png", img_arr)

    # Display or further process the rendered image
    # For example, you can use OpenCV to display the image
    # import cv2
    # cv2.imshow("Rendered Image", img_arr)
    # cv2.waitKey(1)

    # Update the simulation
    p.stepSimulation()
    break

# Close the connection to PyBullet (will not be reached in this script)
p.disconnect()