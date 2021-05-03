# Deep sensor estimation in torcs
Using (mono) image data to guess some "sensor" values from the open source racing game torcs.

The estimated values are:
- track - 19 rangefinder values, fired from the front of the car in the range [-45°,45°]
- angle - angle with the center line
- speedX, speedY, speedZ - speed in all directions
- trackPos - distance from center line

The dataset is created using [pytorcs](https://github.com/gerkone/pyTORCS-docker), is made up of numpy arrays of tuples (sensors, image).

The tested estimator are a plain conv-maxpool network and a ResNet.

Note that this is a very simple experiment and the quality of the code is low.
