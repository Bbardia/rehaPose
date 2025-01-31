# Bio-Feedback
 This code uses pose estimation to calculate the full body joint angle and show it in live time using a camera.

 # Requirements
 This code uses the MMPose package for pose estimation, please follow the setup from their github.
 For the ploting we are using pyqtgraph for a faster and smoother ploting on GPU.

 # Usage
 Run the file Body_Joint.py and it will start the camera and show the live pose estimation, the joint angles will be plotted in real time.
 You can change the model mmpose is using, the original model is vitpos-h, you can change that to any model mmpose supports. 
