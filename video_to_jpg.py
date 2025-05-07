import cv2
import os
import glob

def extract_frames_by_seconds(video_path, output_folder, interval_sec=1):
    os.makedirs(output_folder, exist_ok=True)  # 폴더 없으면 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] {video_path} 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    filename = os.path.splitext(os.path.basename(video_path))[0]

    saved_count = 0
    while True:
        target_sec = saved_count * interval_sec
        target_frame = int(target_sec * fps)

        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        save_path = os.path.join(output_folder, f"{filename}_{int(target_sec):d}s.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1

    cap.release()
    print(f"[완료] {filename}에서 {saved_count}개 프레임 저장됨 (간격: {interval_sec}s)")

def batch_extract(video_folder, output_folder, interval_sec=1):
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))  # mp4만 대상
    print(f"{len(video_files)}개의 비디오 파일을 찾았습니다.")

    for video_path in video_files:
        extract_frames_by_seconds(video_path, output_folder, interval_sec)

# 사용 예시
input_folder = 'videos'         # 비디오 파일들이 있는 폴더
output_folder = 'frames' # 프레임 저장 폴더
interval = 1                    # 1초마다 프레임 추출

batch_extract(input_folder, output_folder, interval_sec=interval)