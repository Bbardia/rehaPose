# Bio-Feedback
 This code uses pose estimation to calculate the full body joint angle and show it in live time using a camera.

 # Requirements
 This code uses the MMPose package for pose estimation, please follow the setup from their [GitHub](https://github.com/open-mmlab/mmpose).
 For the plotting, we are using pyqtgraph for faster and smoother plotting on GPU.
 You can use either the GPU or CPU for the pose estimation by changing the device variable in MMPOSEInfrencer.

 # Usage
 Run the file Body_Joint.py and it will start the camera and show the live pose estimation, the joint angles will be plotted in real-time.
 You can change the model MMpose is using, the original model is vitpos-H, you can change that to any model Mmpose supports. 
 For the best and most accurate result place the camera on the side of the body and make sure the body is in the center of the frame.
