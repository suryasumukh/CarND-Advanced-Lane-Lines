## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/color_threshold1.png ""  
[image2]: ./output_images/color_threshold2.png ""  
[image3]: ./output_images/combined.png ""  
[image4]: ./output_images/combined1.png ""  
[image5]: ./output_images/hls.png ""  
[image6]: ./output_images/hsv.png ""  
[image7]: ./output_images/lab.png ""  
[image8]: ./output_images/lane1.png ""  
[image9]: ./output_images/lane2.png ""  
[image10]: ./output_images/lane_fit.png ""  
[image11]: ./output_images/perspective1.png ""  
[image12]: ./output_images/perspective2.png ""  
[image13]: ./output_images/rgb.png ""  
[image14]: ./output_images/save_output_here.txt ""  
[image15]: ./output_images/sliding_window.png ""  
[image16]: ./output_images/undistorted1.png ""  
[image17]: ./output_images/undistorted2.png ""  
[image18]: ./output_images/unwarped.png ""  
[image19]: ./output_images/warped.png ""  
[image20]: ./output_images/yuv.png ""  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to calibrate camera is in `calibrate.py`. The code is largely referred from the course videos and [udacity/CarND-Camera-Calibration](https://github.com/udacity/CarND-Camera-Calibration) repo.  

First a tensor call "object point" is constructed to represent the corners of the chessboard in real world. This is a 3D tensor with the third dimension z, being 0 and (x, y) represent the inner corners of the chessboard. The calibration images are 10x7 chessboard images and thus the object point will have a dimension of 9x6. `cv2.findChessboardCorners` is used to detect the inner chessboard corners. For every successful detection of 9x6 points the object point is appended to a list of objects points `obj_points` and the corresponding corner points are appended to a list of image points `image_points`.

 `object_points` and `image_points` used to compute the camera calibration and distortion coefficients with `cv2.calibrateCamera()` function.  The images are then undistorted with `cv2.undistort()` function. Here is an example of the original image and undistorted image

![alt text][image16]  

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![alt text][image17]  

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code to generate binary threshold images can be found in `detect.py`. The selection of color schemes and gradient was more of a trial and error. RGB, GRAY, HLS, HSV, YUV schemes were explored along with Sobel gradients, magnitude and direction transforms. Below are some visualization of the schemes.

![alt text][image13]  
![alt text][image5]  
![alt text][image6]  
![alt text][image7]  
![alt text][image20]  

It was found that a combination of Sobel x gradient on S channel from HLS, L and B channels from LAB performed better than
Sobel gray and S channel suggested in the notes. Sobel gradient is wit respect to x-direction with a kernel size of 3 and
threshold between 30 and 100. The B channel has a threshold of [150, 255] and the L channel has a threshold of (200, 255].
  The combined binary is a bitwise-or of the Sobel-sx, L and B channel binaries.

```python
def color_threshold(image):
    S = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    sobel_s = abs_sobel_thresh(S, orient='x', sobel_kernel=3, thresh=(30, 100))

    B = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 2]
    b_binary = np.zeros_like(B)
    b_binary[(B >= 150) & (B <= 255)] = 1

    L = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]
    l_binary = np.zeros_like(L)
    l_binary[(L > 200) & (L <= 255)] = 1

    return l_binary | b_binary | sobel_s
```  

![alt text][image3]  

Combination of L, B and Sobel SX binary images

![alt text][image4]  

The following are examples of color threshold on few images with perspective transform.
![alt text][image1]  
![alt text][image2]  


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform can be found in `detect_lanes` function in `detect.py`. The `src` and `dst` points were chosen by trial and error. The polygon roughly covers the lane are of interest in the image and is based on the assumption that the lane of interest always falls in this region of the image.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 300, 720      |
| 595, 450      | 300, 0        |
| 685, 450      | 1000, 0       |
| 1120, 720     | 1000, 720     |

Here are example of perspective transforms

![alt text][image12]  
![alt text][image11]  

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The detection is mostly performed with histogram and sliding-window techniques suggested in the notes. First an histogram of the lower pat of the image is taken to get a sense of the location of the lane lines. Then a sliding window with a predefined margin is run from the bottom of the image to the top collecting pixel indices that satisfy the predefined pixel density in each window. From the indices, corresponding (x, y) coordinates are determined and a polynomial of order 2 is fit using `np.polyfit` function that approximately describe the location of lanes on the image. The code does not keep tack of previous lane line detection but rather starts again with the histogram and sliding window. The function `find_inital_lines()` can be found in `detect.py`. This function takes a warped image, margin size and pixel density as arguments and return the left and right fit lines.   

![alt text][image19]  
![alt text][image15]  
![alt text][image10]  
![alt text][image18]  

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code can be found in the `draw_lines()` function. The code mostly follows suggestion for course notes.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image8]  
![alt text][image9]  

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Detecting lanes line in shadows and bright light conditions was difficult. After some trial and error with color and thresholding I found a combination of L, B and Sobel S to work but they are not perfect. L channel add a lot of disturbance to the lane lines and not as smooth as Sobel S or B which can be seen in the images above. This can cause fit lines to cross lanes. I only worked with the images from `project_video.mp4`. Lanes were not clearly detected in `challenge_video.mp4`. Also the detection pipeline doesn't store previous detection results. This could help the pipeline perform better.  The pipeline fails if it can't detect lines. Because the pipeline doesn't store history it cannot extrapolate lane lines in image where detection is bad.
