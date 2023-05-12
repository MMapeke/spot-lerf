# SMASH: Spot Maneuvers in A Semantic Habitat

This directory contains additional files we created to interface Boston Dynamics SPOT robot with LERF as a class final for CSCI2952O: A Practical Introduction to Advanced 3D Robot Perception.

# LERF Scripts

We edit the rendering function in the LERF pipeline to calculate the relevancy map for six views, rotated around a virtual origin (reflected in `SMASH/lerf.py/get_outputs_for_camera_ray_bundle()`). Using these views, we find the location of highest relevancy in LERF space, and the rendering function calls `transform()` from `SMASH/xform.py` to transform the point to Spot's local coordinate frame.

# SPOT Navigation Scripts

After connecting SPOT SDK's estop, we can run the navigation script. The navigation script undocks the robot, navigates from the world center (center of the room) back and forth towards goal/object locations, and then undocks at the end. Goal locations are obtained from our LERF scripts that compute a 3D world point with the highest semantic relevancy map between multiple views. Goal locations are entered in the terminal's REPL as "x-coordinate, y-coordinate".
