import os
import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN
from moviepy.editor import VideoFileClip, ImageSequenceClip, ImageClip, concatenate_videoclips
from tqdm import tqdm

def save_interval_frames_to_images(video_clip, output_dir, interval):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in tqdm(enumerate(video_clip.iter_frames())):
        if i % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            Image.fromarray(frame).save(frame_path)

def upscale_images(input_dir, output_dir, upscale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=upscale_factor)
    
    # 根据 upscale_factor 选择相应的权重文件
    if upscale_factor == 2:
        weights_path = 'weights/RealESRGAN_x2.pth'
    elif upscale_factor == 4:
        weights_path = 'weights/RealESRGAN_x4.pth'
    elif upscale_factor == 8:
        weights_path = 'weights/RealESRGAN_x8.pth'
    else:
        raise ValueError("Unsupported upscale factor")
    
    model.load_weights(weights_path, download=True)

    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(os.listdir(input_dir))
    for frame_file in tqdm(frame_files, desc="Upscaling Frames"):
        frame_path = os.path.join(input_dir, frame_file)
        pil_image = Image.open(frame_path)
        with torch.no_grad():
            sr_image = model.predict(pil_image)
        sr_image_path = os.path.join(output_dir, frame_file)
        sr_image.save(sr_image_path)

def merge_images_to_video(input_dir, output_video_path, fps, interval):
    frame_files = sorted(os.listdir(input_dir))
    frames = [os.path.join(input_dir, frame_file) for frame_file in frame_files]
    
    clips = []
    for i, frame_path in enumerate(frames):
        clip = ImageClip(frame_path).set_duration(interval / fps)
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_video_path, fps=fps, codec='libx264')

def upscale_video(input_video_path, output_video_path, upscale_factor=4, interval=10):
    # 加载视频
    video_clip = VideoFileClip(input_video_path)
    fps = video_clip.fps

    # 创建临时目录
    temp_dir_original = "temp_frames_original"
    temp_dir_upscaled = "temp_frames_upscaled"

    # 将间隔帧保存为图片
    save_interval_frames_to_images(video_clip, temp_dir_original, interval)

    # 对所有间隔帧进行上采样
    upscale_images(temp_dir_original, temp_dir_upscaled, upscale_factor)

    # 将上采样后的图片合并成视频
    merge_images_to_video(temp_dir_upscaled, output_video_path, fps, interval)

    # 清除临时文件
    for frame_file in os.listdir(temp_dir_original):
        os.remove(os.path.join(temp_dir_original, frame_file))
    os.rmdir(temp_dir_original)

    for frame_file in os.listdir(temp_dir_upscaled):
        os.remove(os.path.join(temp_dir_upscaled, frame_file))
    os.rmdir(temp_dir_upscaled)

    print(f"Video upscaled and saved to {output_video_path}")

# 示例用法
if __name__ == "__main__":
    input_video_path = "1.mp4"
    output_video_path = "1x4_inter.mp4"
    upscale_factor = 4
    interval = 10  # 每隔 10 帧保存一张图片
    upscale_video(input_video_path, output_video_path, upscale_factor, interval)
