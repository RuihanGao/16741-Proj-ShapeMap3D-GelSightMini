import os
import math 
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict

class UR5Sim():
  
    def __init__(self, model_name):

        # change path to cwd
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)
        pybullet.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=-30, cameraPitch=-90, cameraTargetPosition=[0,0,0])
        pybullet.setGravity(0, 0, -9.8)

        self.robot_path = os.path.join('..','..','..','data', 'misc', 'ure5', 'ur5e_wsg50' + '.urdf') # load dense point cloud
        self.table_path = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
        self.model_path = os.path.join('..','..','..','data', model_name, model_name+'.urdf') # load dense point cloud

        self.ur5 = self.load_scene()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.end_effector_index = 10 # guide_joint_finger_left

        # list controllable joints for UR5
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     

    def load_scene(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(self.table_path, [0, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(self.robot_path, [0.5, 0, 0], [0, 0, 0, 1], flags=flags)
        model = pybullet.loadURDF(self.model_path, [0, 0, 0], useFixedBase=True)
        return robot
    

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = self.contactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            for collision in collisions:
                print("[", collision[0],", ", collision[1], "]")
            return True
        return False

    def contactPoints(self):
            """Gets all contact points and forces
            
            Returns:
                list -- list of entries (link_name, position in m, force in N)
            """
            result = []
            contacts = pybullet.getContactPoints()
            for contact in contacts:
                link_index1 = contact[3]
                link_index2 = contact[4]
                if link_index1 >= 0:
                    link_name1 = (pybullet.getJointInfo(self.ur5, link_index1)[12]).decode()
                else:
                    link_name1 = 'base'
                if link_index2 >= 0:
                    link_name2 = (pybullet.getJointInfo(self.ur5, link_index2)[12]).decode()
                else:
                    link_name2 = 'base'
                result.append((link_name1, link_name2))

            return result 
            
    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))


    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

def demo_simulation():
    """ Demo program showing how to use the sim """
    sim = UR5Sim('mustard_bottle')
    sim.add_gui_sliders()
    while True:
        x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
        joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        sim.set_joint_angles(joint_angles)
        sim.check_collisions()

if __name__ == "__main__":
    demo_simulation()