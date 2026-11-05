import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../models/pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

print("Hello World!")
with PoseLandmarker.create_from_options(options) as landmarker:
    # Synchronous: processes every frame in order with monotonic timestamps (no frame drops)
    for mp_image, timestamp_ms in decoded_video_frames:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        print(result)