import os
import torch.multiprocessing as mp
from video_upscaler_with_skip import *

def batch_upscale_videos(video_paths, upscale_factor=4, max_frames=None, threshold=0, batch_size=6):
    """批量上采样视频"""
    # 将视频路径列表分成每批次6个元素的子列表
    video_batches = [video_paths[i:i + batch_size] for i in range(0, len(video_paths), batch_size)]
    
    for batch in video_batches:
        print(f"Processing batch of {len(batch)} videos...")
        
        # 使用 torch.multiprocessing 并行处理每个批次的视频
        mp.set_start_method('spawn', force=True)
        with mp.Pool() as pool:
            results = []
            for video_path in batch:
                output_video_path = f"{os.path.splitext(video_path)[0]}_x{upscale_factor}.mp4"
                results.append(pool.apply_async(upscale_video, (video_path, output_video_path, upscale_factor, max_frames, threshold)))
            
            # 等待所有任务完成
            for result in results:
                result.get()
        
        print(f"Batch processing complete.")

# 示例视频路径列表
video_paths = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4",
    "video4.mp4",
    "video5.mp4",
    "video6.mp4",
    "video7.mp4",
    "video8.mp4",
    "video9.mp4",
    "video10.mp4",
]

# 批量上采样视频
batch_upscale_videos(video_paths, upscale_factor=4, max_frames=None, threshold=0, batch_size=6)
