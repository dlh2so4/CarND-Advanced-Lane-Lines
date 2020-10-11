**Advanced Lane Finding Project**

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

[image1]: ./output_images/test1_undistorted.png "Undistorted"
[image2]: ./output_images/image_undistort.jpg "Road Transformed"
[image3]: ./output_images/image_edge.jpg "Binary Example"
[image4]: ./output_images/image_gray.png "Warp Example"
[image5]: ./output_images/lane_line_poly.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced Detection.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
I wrote a function called undistortImg(img, mtx, dist), img is the original image, mtx and dist are the output from the camera calibration process.
Inside the function, simply apply the openCV undistort function, I get the undistorted image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, the function is called edgeDetection and it is in the 3rd code cell of Advanced Detection.ipynb.  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warpImg()`, which appears in the 3rd code cell of Advanced Detection.ipynb.  The `warpImg()` function takes as inputs an image (`img`), as well as Y(Y/1*image height is the Y axis location of the upper boundary of the source area), and X(X/1*image width is the width of the upper boundary fo the source area), with these input, I calculate the source and destination  of the perspective transform and then warp the image with warp function:

```python
def warpImg(img,X,Y):

    src = np.float32([[(1-X)/2*img.shape[1],Y*img.shape[0]],[0,img.shape[0]],[img.shape[1],img.shape[0]],[(1+X)/2*img.shape[1],Y*img.shape[0]]])
    
    dst = np.float32([[0,0],[0,img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1],0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M
```
The image below shows the warp result

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I convert the warpped image to gray scale and start the lane line detection process. The main pipeline is in the 4th code cell of Advanced Detection.ipynb, called process_image.

My method comes with a state machine, includes two states: Normal Mode and Init Mode.

    When we have lane line detection result from previous frame, we run Normal Mode and detect within a certain width around the previous lane curve.
    When we initialize the detection or when we lost lane lines for concecutive n(20 in this case) frames, we run Init Mode.

    In Init Mode, we use the slide window method to detect left and right lane line pixels and then use the numpy polyfit function to fit the detected pixels with a 2nd order polynomial.

    In Normal Mode, we directly find pixels around the previous known lane lines and then use polyfit function to get the polynomial coefficients.

I also use a smooth function to remember and calculate the average value of the last n coefficients, so that I get a very smooth curve rather than a unstable lane line.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function measure_curvature_real() and laneCenterCalc(). They are in the 3rd code cell of Advanced Detection.ipynb.
    For the function measure_curvature_real(ploty, left_fit_cr, right_fit_cr): ploty is the y values of the fitted lane line pixels, left_fit_cr and right_fit_cr represent the left and right lane line polynomial coefficients.
    Radius of curvature is R = (1+(2*A*y+B)^2)^1.5/abs(2*A), with this equation, I am able to calculate the lane line curvature for both left and right line, then I take the average of them to indicate the lane curvature.
    
    For the function laneCenterCalc(left_fitx, right_fitx, imgWidth): left_fitx and right_fitx are the fitted lane pixels x values of left and right lane line. imgWidth is the width of the warpped image.
    To calculate the offset between the vehicle center and lane center, I assueme the center of the image is the center of the vehicle. Then I take the 100 x values(the 100 x values close to the bottom of the image, which is close to the vehicle in real world) of the left and right lane line and calcualte the mean x for left and right line.
    The average of left x and right x is the center of the lane and simply divide it with the x value for the center of the image, which is half of the image width.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I used the camera calibration technique to get the factors needed to undistort the image to get an accurate image. Then I combined the gradient and color approach for edge detection. Then I warp the image and conver it to gray scale for detailed lane line detection. A slide window is used to initialize and reinitialize the detection. Then the polynomial coefficients are used to define an area in the new frame for faster lane piexl detection. Once the dection is done, I calculate the curvature of the lane lines and save all the lane line information in a class called Line(). I also check the detection results to see if the lane line are reasonable, the checked information include left and right line should have similiar curvature and roughly parallel. I also make sure the lane line distance between the two lines is reasonable.

How ever, this method need to detect the lane line when there is no lane line-like noise, for the challenge video, it is not able to distinguish the lane line with the dark road seal and road curb.

Hence, the algorithm should implement a method to check the lane line pixel color after the detection to make sure the "lane line" are really lane lines. If they are not, the algorithm should be able to avoid this wrong line and detect other pixels.
