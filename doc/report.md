


Transition from InceptionV3 to Nvidia--better performance with only a little data and a few epochs--maybe Inception makes many features useful for classification but redundant or useless for driving.


Validation split must be nonrandom.

Cropping to center region helps with consistency.

l2 regularization, even in very small amounts, results in understeering. Maybe if combined with data-balancing?

Leaving out zero-turning-angle samples (and their associated side camera views) with probability 0.93 make the model take turns much more reliably. I can now get up to 30 MPH, though not without some curb-hits.