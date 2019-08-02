# circle-detection

You can use the FindCircle function in CircleDetector class.
It will return the CircleF of highest score among the circles found with your estimated radius and error rate.

Example: You wanna find the circle with 100(+-40) px of Radius (the default error value is 30):
  ```c#
  CircleF circle = FindCircle(img, estimatedRadius: 100,error: 40);
  ```
