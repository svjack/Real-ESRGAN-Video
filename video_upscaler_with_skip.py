#### 只有跳过逻辑，没有插值逻辑，threshold > 0 有风险
import os
import torch
from PIL import Image, ImageChops, ImageStat
from py_real_esrgan.model import RealESRGAN
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import time
import uuid
import shutil

def upscale_images(input_dir, output_dir, upscale_factor=4):
    # 根据 upscale_factor 选择相应的权重文件
    if upscale_factor == 2:
        weights_path = 'weights/RealESRGAN_x2.pth'
    elif upscale_factor == 4:
        weights_path = 'weights/RealESRGAN_x4.pth'
    elif upscale_factor == 8:
        weights_path = 'weights/RealESRGAN_x8.pth'
    else:
        raise ValueError("Unsupported upscale factor")

    # 初始化 RealESRGAN 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=upscale_factor)
    model.load_weights(weights_path, download=True)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录下的所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 逐张图片上采样
    for image_file in tqdm(image_files, desc="Upscaling Images"):
        # 读取图片
        image_path = os.path.join(input_dir, image_file)
        pil_image = Image.open(image_path)

        # 使用 RealESRGAN 进行上采样
        with torch.no_grad():
            sr_image = model.predict(pil_image)

        # 生成新的图片名称，并添加后缀
        base_name, ext = os.path.splitext(image_file)
        if upscale_factor == 2:
            new_image_name = f"{base_name}_x2{ext}"
        elif upscale_factor == 4:
            new_image_name = f"{base_name}_x4{ext}"
        elif upscale_factor == 8:
            new_image_name = f"{base_name}_x8{ext}"

        # 保存上采样后的图片到输出目录
        sr_image_path = os.path.join(output_dir, new_image_name)
        sr_image.save(sr_image_path)

    print(f"All images upscaled and saved to {output_dir}")

def get_image_diff(image0: Image.Image, image1: Image.Image) -> float:
    """计算两张图片之间的差异百分比"""
    difference_stat = ImageStat.Stat(ImageChops.difference(image0, image1))
    return sum(difference_stat.mean) / (len(difference_stat.mean) * 255) * 100

def upscale_video(input_video_path, output_video_path, upscale_factor=4, max_frames=None, threshold=0):
    # 根据 upscale_factor 选择相应的权重文件
    if upscale_factor == 2:
        weights_path = 'weights/RealESRGAN_x2.pth'
    elif upscale_factor == 4:
        weights_path = 'weights/RealESRGAN_x4.pth'
    elif upscale_factor == 8:
        weights_path = 'weights/RealESRGAN_x8.pth'
    else:
        raise ValueError("Unsupported upscale factor")

    # 初始化 RealESRGAN 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=upscale_factor)
    model.load_weights(weights_path, download=True)

    # 加载视频
    video_clip = VideoFileClip(input_video_path)
    original_fps = video_clip.fps  # 获取原视频的帧率

    # 创建一个随机名称的临时目录来存储上采样后的帧
    temp_dir = f"temp_frames_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    # 逐帧上采样
    frames = []
    frame_durations = []
    total_frames = int(video_clip.fps * video_clip.duration)
    previous_frame = None
    for i, frame in enumerate(tqdm(video_clip.iter_frames(), total=total_frames, desc="Upscaling Frames")):
        if max_frames is not None and i >= max_frames:
            break
        
        # 将帧转换为 PIL 图像
        pil_image = Image.fromarray(frame)
        
        # 如果阈值大于0，跳过相似的相邻帧
        if threshold > 0 and previous_frame is not None:
            difference_ratio = get_image_diff(previous_frame, pil_image)
            if difference_ratio < threshold:
                # 跳过当前帧，使用前一帧的结果
                frames.append(frames[-1])
                frame_durations.append(1 / original_fps)
                continue
        
        # 使用 RealESRGAN 进行上采样
        with torch.no_grad():
            sr_image = model.predict(pil_image)
        
        # 将上采样后的图像保存到临时目录
        sr_image_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
        sr_image.save(sr_image_path)
        frames.append(sr_image_path)
        frame_durations.append(1 / original_fps)  # 每个帧的持续时间
        
        # 更新 previous_frame
        previous_frame = pil_image

    # 从上采样后的帧创建新的视频
    sr_video_clip = ImageSequenceClip(frames, durations=frame_durations)
    
    # 检查原视频是否有声音，并将其添加到生成后的视频中
    if video_clip.audio is not None:
        sr_video_clip = sr_video_clip.set_audio(video_clip.audio)
    
    # 设置帧率
    sr_video_clip.fps = original_fps
    
    sr_video_clip.write_videofile(output_video_path, codec='libx264')
    shutil.rmtree(temp_dir)

    print(f"Video upscaled and saved to {output_video_path}")

if __name__ == "__main__":
    upscale_images("pic_unzip_save_dir_pinyin_crop_rm_bk_merge_crop_zh", "IP-Adapter原神角色图片(face-crop-plus-chinese)-LivePortrait测试-v2-x2", 2)

    upscale_images("pic_unzip_save_dir_pinyin_crop_rm_bk_merge_crop_zh", "IP-Adapter原神角色图片(face-crop-plus-chinese)-LivePortrait测试-v2-x4", 4)
    
    input_video_path = "丽莎动态.mp4"
    output_video_path = "丽莎动态_skp2x2.mp4"
    upscale_factor = 2
    threshold = 2  # 设置阈值，0表示不跳过任何帧
    upscale_video(input_video_path, output_video_path, upscale_factor, max_frames=None, threshold=threshold)
    
    input_video_path = "丽莎.mp4"
    output_video_path = "丽莎_skp2x05.mp4"
    upscale_factor = 2
    threshold = 0.5  # 设置阈值，0表示不跳过任何帧
    upscale_video(input_video_path, output_video_path, upscale_factor, max_frames=None, threshold=threshold)
