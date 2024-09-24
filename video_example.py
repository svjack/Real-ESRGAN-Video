import os
import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm

def upscale_video(input_video_path, output_video_path, upscale_factor=4, max_frames=None):
    # 初始化 RealESRGAN 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=upscale_factor)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    # 加载视频
    video_clip = VideoFileClip(input_video_path)

    # 创建一个临时目录来存储上采样后的帧
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # 逐帧上采样
    frames = []
    total_frames = int(video_clip.fps * video_clip.duration)
    for i, frame in enumerate(tqdm(video_clip.iter_frames(), total=total_frames, desc="Upscaling Frames")):
        if max_frames is not None and i >= max_frames:
            break
        
        # 将帧转换为 PIL 图像
        pil_image = Image.fromarray(frame)
        
        # 使用 RealESRGAN 进行上采样
        with torch.no_grad():
            sr_image = model.predict(pil_image)
        
        # 将上采样后的图像保存到临时目录
        sr_image_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
        sr_image.save(sr_image_path)
        frames.append(sr_image_path)

    # 从上采样后的帧创建新的视频
    sr_video_clip = ImageSequenceClip(frames, fps=video_clip.fps)
    sr_video_clip.write_videofile(output_video_path, codec='libx264')

    # 清除临时文件
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(temp_dir)

    print(f"Video upscaled and saved to {output_video_path}")

if __name__ == "__main__":
  input_video_path = "1.mp4"
  output_video_path = "1x4.mp4"
  upscale_factor = 4
  upscale_video(input_video_path, output_video_path, upscale_factor, max_frames=10)
