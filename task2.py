import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pose=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_z= init_pose[2]

        # Goal
        self.target_pose = target_pose if target_pose is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        """Takeoff"""
       # punish_x = np.tanh(abs(self.sim.pose[0] - self.target_pose[0]))
        #punish_y = np.tanh(abs(self.sim.pose[1] - self.target_pose[1]))
        
    
        reward_z = 3*np.tanh(self.sim.pose[2] - self.init_z)
        
        punish_rot1 = np.tanh(abs(self.sim.pose[3]))
        punish_rot2 = np.tanh(abs(self.sim.pose[4]))
        punish_rot3 = np.tanh(abs(self.sim.pose[5]))
        
        reward = reward_z  - punish_rot1 - punish_rot2 - punish_rot3
        
        dist = abs(np.linalg.norm(self.target_pose[:3] - self.sim.pose[:3]))
        
        if dist < 3:
            reward += 10*np.tanh(dist)
        else:
            reward -= np.tanh(dist)
        
        reward-=np.tanh(np.linalg.norm(self.sim.v))
        reward-=np.tanh(np.linalg.norm(self.sim.angular_v))
        
        if self.sim.v[2] > 0:
            reward+=np.tanh(self.sim.v[2])
        
        return reward
            

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state