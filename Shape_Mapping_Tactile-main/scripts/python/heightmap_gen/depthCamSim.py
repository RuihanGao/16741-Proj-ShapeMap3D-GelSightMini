# input the set of poses and object file, and generate the ground truth heightmaps, tactile images and contact masks
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os
from os import path as osp
import sys
import trimesh
import pyrender
# import open3d as o3d
import pdb

class depthCamSim:

    def __init__(self, obj_path, save_path):
        # dense point cloud with vertice information
        fuze_trimesh = trimesh.load(obj_path)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        obj_center = mesh.centroid
        
        # MATLAB view(3)
        T_wc = np.zeros((4,4)) # transform from point coord to world coord
        T_wc[3,3] = 1.0
        T_wc[0:3,3] = np.array([0.5 + obj_center[0], 0.0 + obj_center[1], 0.5]) # translation
        s = np.sqrt(2)/2
        T_wc[0:3,0:3] = np.array([[0.0, -s, s],   
                                  [1.0, 0.0, 0.0],   
                                  [0.0, s, s]])   # rotation
        
        # print(T_wc)
        scene.add(camera, pose=T_wc)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        scene.add(light, pose=T_wc)
        r = pyrender.OffscreenRenderer(400, 400)
        color, depth = r.render(scene)
        fig = plt.figure()
        ax = plt.subplot(1,3,1)
        # ax.set_title("Image rendering", size=10)
        plt.axis('off')
        plt.imshow(color)
        ax = plt.subplot(1,3,2)
        # ax.set_title("Depth image", size=10)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)


        pc = self.pointcloud(depth, np.pi / 3.0)
        mu, sigma = 0, 0.001

        step_size = int(np.ceil(pc.shape[0]/500))
        # pc = pc[0::step_size,:]
        noise = np.random.normal(mu, sigma, size=(pc.shape[0], 3))

        pc[:, 0:3] +=  noise
        # T_cw = np.linalg.inv(T_wc) # transform from world coord to point coord

        print('depth cloud size: ', pc.shape[0])
        tf_pc = np.dot(T_wc, pc.T).T[:,0:3] # origin is the center of image plane, N * 3

        ax = plt.subplot(1, 3, 3, projection='3d')
        # ax.set_title("Point cloud", size=10)
        ax.scatter3D(tf_pc[:, 0], tf_pc[:, 1], tf_pc[:, 2], c=tf_pc[:, 2], cmap='viridis', s = .1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.axis('off')

        ax.set_xlim(-0.1 + obj_center[0], 0.5 + obj_center[0])
        ax.set_ylim(-0.25 + obj_center[1], 0.25 + obj_center[1])
        ax.set_zlim(0.0, 0.5)
        self.plotCoordinateFrame(ax, T_wc, size=.05, linewidth=2)
        fig.tight_layout()

        np.save(osp.join(save_path, 'depthCam.npy'), tf_pc)
        fig.savefig(osp.join(save_path, 'depthCam.pdf'), dpi=1500, bbox_inches = 'tight', pad_inches = 0)
        plt.show(block=False)
        plt.pause(10)
        plt.close()

    # https://github.com/mmatl/pyrender/issues/14#issuecomment-485881479
    def pointcloud(self, depth, fov):
        fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)
        
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        
        normalized_y = -normalized_y

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        # world_z = depth[y, x]
        world_z = -depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T


    # https://github.com/ethz-asl/kalibr/blob/master/Schweizer-Messer/sm_python/python/sm/plotCoordinateFrame.py
    def plotCoordinateFrame(self, axis, T_0f, size=1, linewidth=3):
        """Plot a coordinate frame on a 3d axis. In the resulting plot,
        x = red, y = green, z = blue.
        
        plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)
        Arguments:
        axis: an axis of type matplotlib.axes.Axes3D
        T_0f: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
        size: the length of each line in the coordinate frame
        linewidth: the width of each line in the coordinate frame
        Usage is a bit irritating:
        import mpl_toolkits.mplot3d.axes3d as p3
        import pylab as pl
        f1 = pl.figure(1)
        # old syntax
        # a3d = p3.Axes3D(f1)
        # new syntax
        a3d = f1.add_subplot(111, projection='3d')
        # ... Fill in T_0f, the 4x4 transformation matrix
        plotCoordinateFrame(a3d, T_0f)
        see http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html for more details
        """
        # \todo fix this check.
        #if type(axis) != axes.Axes3D:
        #    raise TypeError("axis argument is the wrong type. Expected matplotlib.axes.Axes3D, got %s" % (type(axis)))

        p_f = np.array([ [ 0,0,0,1], [size,0,0,1], [0,-size,0,1], [0,0,-size,1]]).T;
        p_0 = np.dot(T_0f,p_f)
        # X-axis

        X = np.append( [p_0[:,0].T] , [p_0[:,1].T], axis=0 )
        Y = np.append( [p_0[:,0].T] , [p_0[:,2].T], axis=0 )
        Z = np.append( [p_0[:,0].T] , [p_0[:,3].T], axis=0 )
        axis.plot3D(X[:,0],X[:,1],X[:,2],'r-', linewidth=linewidth)
        axis.plot3D(Y[:,0],Y[:,1],Y[:,2],'g-', linewidth=linewidth)
        axis.plot3D(Z[:,0],Z[:,1],Z[:,2],'b-', linewidth=linewidth)
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        obj = str(sys.argv[1]) # 'mustard_bottle', 'power_drill'
        touchFile = str(sys.argv[2]) # 'mustard_bottle', 'power_drill'
    else:
        obj = "005_tomato_soup_can"
        touchFile = "textured_60sampled.mat"

    # change path to cwd
    abspath = osp.abspath(__file__)
    dname = osp.dirname(abspath)
    os.chdir(dname)

    print('Rendering depth camera: ' + obj)
    obj_path = osp.join('..','..','..','models', obj, 'google_512k', 'textured.obj') # load dense point cloud
    save_path = osp.join('..','..','..','gelsight_data', os.path.splitext(touchFile)[0], obj)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    generator = depthCamSim(obj_path, save_path)
