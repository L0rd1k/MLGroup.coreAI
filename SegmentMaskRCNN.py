from pixellib.instance import instance_segmentation
import mediapipe as mp
import cv2
import numpy as np

def mainPoseEstimatorExecutor():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture("D:\Video\Chaplin.mp4")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Pose Estimator", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()  

def mainSegmentationExecutor():
    segmentation_model = instance_segmentation()
    segmentation_model.load_model("D:\Video\mask_rcnn_coco.h5")
    cap = cv2.VideoCapture("D:\Video\Chaplin.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        res = segmentation_model.segmentFrame(frame, show_bboxes=True)
        image = res[1]
        cv2.imshow("Instance Segmentation", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    mainPoseEstimatorExecutor()
    # mainSegmentationExecutor()

if __name__ == "__main__":
    main()