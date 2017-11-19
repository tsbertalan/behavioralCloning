


Transition from InceptionV3 to Nvidia--better performance with only a little data and a few epochs--maybe Inception makes many features useful for classification but redundant or useless for driving.


Validation split must be nonrandom.

Cropping to center region helps with consistency.

l2 regularization, even in very small amounts, results in understeering. Maybe if combined with data-balancing?

Leaving out zero-turning-angle samples (and their associated side camera views) with probability 0.93 make the model take turns much more reliably. I can now get up to 30 MPH, though not without some curb-hits.

I added dynamic control of speed.
Obviously, it would be better to anticipate curves, and slow down in advance, but that would require a different model. This is at least a little better than a static speed.

Discovered that training on just 10% of a smaller dataset (just `mouseForward.zip`), for only 1 epoch results in less overfitting and smoother driving. Combined with the dynamic speed control, this is very reliable.

It also works passably using only the provided data.

Setting sidecamAdjustment to .9 results in sharp flailing (like a too-large kp on a p-controller). Setting it to .01 results in running off the road. One way to determine the "correct" value would be to find the center image most similar to each side image, then choose the sidecamAdjustment value that brings the corresponding steering angles into closest agreement. This would depend on the measure of closeness--one possibility is to take a previously trained network, and forward-evaluate the final flattened feature vector after the convolutional layers. "Closest" could then be taken in the Euclidean sense in this space.

Another possibility is to observe the control actions suggested for a known smooth curve--in this sequene of plots [doc/sidecamstrength], somewhere between .1 and .2 is the dividing line where the controller starts producing actual negative values rather than only zeros (corresponding to a lack of rightward turn torque, and therefore appropriate for maintaining a gentle rightward turn). I settled on .15.

Interestingly, this insight provides most of what is needed to get some sort of results on the challenge track. There, much higher values of sidecamAdjustment are required to emphasie the much larger need for corrective steering when off centerline. Additionally, a lower target maximum speed helps, as well as a more aggressive back-off on speed when turning is detected.

Occasionally on the main track, and frequently on the challenge track, the wheels would lock up and the car would not move until the brakes were tapped manually, despite nonzero speed setpoint. This may have been bug in the physics, or may have been a bug in the PI controller (I have modified the controller to zero the integral term when brakes or gas is tapped, to short-circuit integral windup). I didn't investigate closely enough to determine.