import os
import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def upscale_batch(frames, upscale_factor):
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

    sr_images = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        with torch.no_grad():
            sr_image = model.predict(pil_image)
        sr_images.append(sr_image)
    return sr_images

def upscale_video(input_video_path, output_video_path, upscale_factor=4, max_frames=None, num_workers=4, batch_size=8):
    # 加载视频
    video_clip = VideoFileClip(input_video_path)

    # 创建一个临时目录来存储上采样后的帧
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # 逐帧上采样
    frames = []
    total_frames = int(video_clip.fps * video_clip.duration)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        batch = []
        for i, frame in enumerate(video_clip.iter_frames()):
            if max_frames is not None and i >= max_frames:
                break
            batch.append(frame)
            if len(batch) >= batch_size:
                futures.append(executor.submit(upscale_batch, batch, upscale_factor))
                batch = []
        if batch:
            futures.append(executor.submit(upscale_batch, batch, upscale_factor))

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Upscaling Frames")):
            sr_images = future.result()
            for j, sr_image in enumerate(sr_images):
                sr_image_path = os.path.join(temp_dir, f"frame_{i * batch_size + j:06d}.png")
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

# 示例用法
if __name__ == "__main__":
    mp.set_start_method('spawn')
    input_video_path = "1.mp4"
    output_video_path = "1x4_batch.mp4"
    upscale_factor = 4
    max_frames = None  # 只处理前 100 帧
    num_workers = 1  # 使用 4 个并行工作进程
    batch_size = 4  # 每个批次处理 8 帧
    upscale_video(input_video_path, output_video_path, upscale_factor, max_frames, num_workers, batch_size)
