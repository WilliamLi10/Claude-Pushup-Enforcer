import mediapipe as mp
import numpy as np
import cv2 as cv
import time
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

def main():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="./models/pose_landmarker_lite.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )

    print("Hello World!")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return


    with PoseLandmarker.create_from_options(options) as landmarker:
        # Synchronous: processes every frame in order with monotonic timestamps (no frame drops)
        while True:

            ret, frame = cap.read()
            frame = np.array([[pixel[::-1] for pixel in row] for row in frame])
            img_frame = mp.Image(mp.ImageFormat.SRGB, frame)
            if not ret:
                print("can not process frame")
                return 
            
            result = landmarker.detect_for_video(img_frame, int(time.monotonic() * 1000))
            transformedFrame = draw_landmarks_on_image(frame, result)

            cv.imshow('frame', transformedFrame)
            if cv.waitKey(1) == ord('q'):
                break

            if cv.getWindowProperty('frame', 4) == 0:
                break
        cap.release()
        cv.destroyAllWindows()





if __name__ == '__main__':
    main()