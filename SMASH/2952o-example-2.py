# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command the robot to go to an offset position using a trajectory command."""

import logging
import math
import sys
import time
import numpy as np
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import basic_command_pb2
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers, robot_command
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.docking import blocking_dock_robot, blocking_undock, get_dock_id

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client import robot_command
from bosdyn.client.docking import blocking_dock_robot, blocking_undock, get_dock_id


_LOGGER = logging.getLogger(__name__)

# Code uses pieces from SPOT SDk dock_my_robot and frame_trajectory_command tutorials
# Before running this code, connect estop

# Example queries and world locations from our demo for 2952O
# spider = [2.3 -5.4]
# music = [1, -3.4]
# cat = [3.2, -1.8]
# lemon = [3.5, -3.2]
# walle = [1.66, -5.4]
# hat = [3.2, -4.2]
# coffee = [-0.5, -3.18]



def main():

    options = {
        'dx' : 0.0,
        'dy' : 0.0,
        'dyaw' : 0.0,
        'frame' : ODOM_FRAME_NAME, 
        'stairs' : False,
        'dock-id' : 520,
    }

    bosdyn.client.util.setup_logging(False)

    sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
    robot = sdk.create_robot('tusker.rlab.cs.brown.edu')
    bosdyn.client.util.authenticate(robot)

    # Check that an estop is connected with the robot so that the robot commands can be executed.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Create the lease client.
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    # Setup clients for the robot state and robot command services.
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.time_sync.wait_for_sync()
        robot.power_on()

        blocking_stand(robot_command_client)

        # TODO: Walk to center of scene
        world_center = np.array([1.5, -2])
        options['dx'] = world_center[0]
        options['dy'] = world_center[1]
        options['dyaw'] = 0

        spot_relative_move(options, robot_command_client, robot_state_client)

        # START LOOP here
        while True :
            # TODO: Move to object goal, world space

            raw_input = input("Input world x,y. Ex. '0.0 -3.0'. No input to return spot to dock \n\n")

            raw_input_list = raw_input.split()

            if len(raw_input_list) == 0 :
                break
            if len(raw_input_list) != 2 :
                print("Enter 2 coordinates; Error: List len != 2")
                continue

            goal = np.array([float(raw_input_list[0]), float(raw_input_list[1])])
            # goal = np.array([0, -3]) # Should scale down by a little
            v = goal - world_center
            v = 0.75 * v # Scale factor, so SPOT doesn't hit object

            rotation = vector_to_degrees(v)
            
            options['dx'] = v[0]
            options['dy'] = v[1]
            options['dyaw'] = rotation

            spot_relative_move(options, robot_command_client, robot_state_client)

            time.sleep(3)
            # TODO: Move back to center and then rotate

            options['dx'] = -1 * np.sqrt(options['dx']**2 + options['dy']**2)
            options['dy'] = 0
            options['dyaw'] = -1 * options['dyaw'] 

            spot_relative_move(options, robot_command_client, robot_state_client)

        # TODO: Rotate to dock, Dock
        options['dx'] = 0
        options['dy'] = 0

        rotation = vector_to_degrees(world_center)

        options['dyaw'] = rotation

        spot_relative_move(options, robot_command_client, robot_state_client)

        # Stand before trying to dock.
        robot_command.blocking_stand(robot_command_client)
        blocking_dock_robot(robot, options['dock-id'])
        print("Docking Success")

def vector_to_degrees(v):
                x, y = v
                angle = np.degrees(np.arctan2(y, x))
                return angle

# Moves spot based on options params
def spot_relative_move(options, robot_command_client, robot_state_client) : 
    # Power on the robot and stand it up.
    relative_move(options['dx'], options['dy'], math.radians(options['dyaw']), options['frame'],
                        robot_command_client, robot_state_client, stairs=options['stairs'])
    
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())

def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("Failed to reach the goal")
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            time.sleep(1)
            return True
        

if __name__ == "__main__":
    if not main():
        sys.exit(1)
