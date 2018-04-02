# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//Term1/FindingLaneLines//test_images_result//solidYellowCurve.jpg]: # (Image References)


---


My pipeline consisted of 4 steps. First, I converted the images to grayscale, then I applied canny edge detection algorithm to detect the edges of the lines in the image, then I applied the hough transform and detected the lane lines in the image and then I draw the line on the lane line detected images. 

