# SMASH: Spot Maneuvers in A Semantic Habitat

This directory contains additional files we created to interface Boston Dynamics SPOT robot with LERF as a class final for CSCI2952O: A Practical Introduction to Advanced 3D Robot Perception.

[Final Report](https://drive.google.com/file/d/1xzpe-a9Lp_ccslDDMO_4urT6rv3kd2wU/view?usp=sharing)
[Short Demo Video](https://drive.google.com/file/d/11go2VWBHB1cKedp771yEiJ3S1b1Lqeq1/view?usp=sharing)
[Long Demo Video](https://drive.google.com/file/d/1F2oHAlvXLwsTwE7wVSbFzxwyRlOsEyvQ/view?usp=sharing)
[Proposal Presentation](https://docs.google.com/presentation/d/1mHz9nTIwlaCL6eMmy-DZaA4OrZ_4qwHNL4Ozt52mhl8/edit?usp=sharing)
[Midpoint Presentation](https://docs.google.com/presentation/d/1vF-sz-a6Kxjesg191bDGCvG-Y6Au9MacST7yZ01y-Xs/edit?usp=sharing)

# LERF Scripts

We edit the rendering function in the LERF pipeline to calculate the relevancy map for six views, rotated around a virtual origin (reflected in `SMASH/lerf.py/get_outputs_for_camera_ray_bundle()`). Using these views, we find the location of highest relevancy in LERF space, and the rendering function calls `transform()` from `SMASH/xform.py` to transform the point to Spot's local coordinate frame.

# SPOT Navigation Scripts

After connecting SPOT SDK's estop, we can run the navigation script. The navigation script undocks the robot, navigates from the world center (center of the room) back and forth towards goal/object locations, and then undocks at the end. Goal locations are obtained from our LERF scripts that compute a 3D world point with the highest semantic relevancy map between multiple views. Goal locations are entered in the terminal's REPL as "x-coordinate, y-coordinate".
