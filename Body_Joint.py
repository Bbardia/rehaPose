import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

# ---------------------------------
# 1. Configuration
# ---------------------------------

# Joints to track
ALL_JOINTS = [
    "left_knee", "right_knee",
    "left_hip",  "right_hip",
    "left_elbow","right_elbow",
    "left_shoulder","right_shoulder"
]

# Keypoint triplets for each joint (COCO format)
JOINT_KEYPOINTS = {
    "left_knee":       ("left_hip",      "left_knee",   "left_ankle"),
    "right_knee":      ("right_hip",     "right_knee",  "right_ankle"),
    "left_hip":        ("left_shoulder", "left_hip",    "left_knee"),
    "right_hip":       ("right_shoulder","right_hip",   "right_knee"),
    "left_elbow":      ("left_shoulder", "left_elbow",  "left_wrist"),
    "right_elbow":     ("right_shoulder","right_elbow", "right_wrist"),
    "left_shoulder":   ("left_hip",      "left_shoulder","left_elbow"),
    "right_shoulder":  ("right_hip",     "right_shoulder","right_elbow"),
}

# COCO keypoint indices
COCO_INDICES = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

# Skeleton for visualization
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

# Rolling window size for PyQtGraph
window_size = 60  # Adjusted window size

# ---------------------------------
# 2. Utility Functions
# ---------------------------------

def calculate_angle(a, b, c):
    """
    Returns the angle at point b formed by points a-b-c.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec1 = a - b
    vec2 = c - b
    
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom < 1e-6:
        return 0.0  # Avoid invalid calculations

    angle = np.arccos(
        np.clip(np.dot(vec1, vec2) / denom, -1.0, 1.0)
    )
    return np.degrees(angle)

def draw_keypoints_and_skeleton(frame, keypoints):
    """
    Draw detected keypoints and skeleton on the frame.
    """
    # Draw keypoints
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Draw skeleton
    for start, end in SKELETON:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def update_plot(joint_name, frame_idx, live_angle):
    """
    Updates the PyQtGraph live angle plot for the given joint.
    """
    global plots, live_angles_history, frame_indices

    live_angles_history[joint_name].append(live_angle)
    frame_indices[joint_name].append(frame_idx)

    # Keep data within the window size
    if len(frame_indices[joint_name]) > window_size:
        live_angles_history[joint_name] = live_angles_history[joint_name][-window_size:]
        frame_indices[joint_name] = frame_indices[joint_name][-window_size:]

    # Update the plot
    plots[joint_name]['live'].setData(frame_indices[joint_name], live_angles_history[joint_name])
    plots[joint_name]['plot_widget'].setXRange(frame_indices[joint_name][0], frame_indices[joint_name][-1])

# ---------------------------------
# 3. PyQtGraph Setup (2 Columns)
# ---------------------------------

pg.setConfigOption('background', 'w')

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Angles Plot")
win.resize(1200, 800)  # Increased window height to fit all plots
win.setWindowTitle('Live Joint Angles')

plots = {}
live_angles_history = {}
frame_indices = {}

# Create 2-column layout (4 rows x 2 columns)
cols = 2
rows = 4

for i, joint in enumerate(ALL_JOINTS):
    if i % cols == 0 and i != 0:
        win.nextRow()  # Move to next row after every 2 columns
    
    plot_widget = win.addPlot(title=joint)
    plot_widget.setLabel('left', 'Angle', units='deg')
    plot_widget.showGrid(x=True, y=True)

    # Create live angle curve (blue)
    live_curve = plot_widget.plot(pen='b', name='Your Angle')

    plots[joint] = {
        'plot_widget': plot_widget,
        'live': live_curve
    }

    live_angles_history[joint] = []
    frame_indices[joint] = []

# ---------------------------------
# 4. MMPose Inference + Webcam
# ---------------------------------

inferencer = MMPoseInferencer(pose2d='vitpose-b', device='cuda')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Live Video', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Live Video', 1280, 720)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam. Retrying...")
        continue

    # Pose inference
    results = inferencer(frame, flip_test=True)

    for res in results:
        keypoints = res["predictions"][0][0]["keypoints"]

        # Draw skeleton on the frame
        draw_keypoints_and_skeleton(frame, keypoints)

        # Calculate angles for all joints
        for joint in ALL_JOINTS:
            kpA, kpB, kpC = JOINT_KEYPOINTS[joint]
            iA = COCO_INDICES[kpA]
            iB = COCO_INDICES[kpB]
            iC = COCO_INDICES[kpC]

            live_angle = calculate_angle(keypoints[iA], keypoints[iB], keypoints[iC])

            # Update live plot
            update_plot(joint, frame_idx, live_angle)

    # Show webcam feed
    cv2.imshow('Live Video', frame)

    # Update PyQtGraph
    QtWidgets.QApplication.processEvents()

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
