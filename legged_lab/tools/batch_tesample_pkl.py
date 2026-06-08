import argparse
import pickle
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm  # 如果提示找不到模块，请在终端运行: pip install tqdm

def interpolate_linear(old_times, old_data, new_times):
    """对多维数组（位置、速度、key_body_pos）进行线性插值"""
    if old_data is None: return None
    interpolator = interp1d(old_times, old_data, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_times).astype(np.float32)

def interpolate_quaternion(old_times, old_quats_xyzw, new_times):
    """对四元数进行球面线性插值 (Slerp)，保证旋转平滑不变形"""
    if old_quats_xyzw is None: return None
    # 归一化四元数
    norms = np.linalg.norm(old_quats_xyzw, axis=1, keepdims=True)
    old_quats_xyzw = old_quats_xyzw / np.clip(norms, 1e-10, None)
    
    rotations = R.from_quat(old_quats_xyzw)
    slerp = Slerp(old_times, rotations)
    new_rotations = slerp(new_times)
    return new_rotations.as_quat().astype(np.float32)

def process_single_file(input_path, output_path, target_fps, input_fps_override=None):
    """处理单个文件的核心流水线"""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # 1. 确定原始 FPS
    if input_fps_override is not None:
        original_fps = float(input_fps_override)
    else:
        original_fps = data.get('fps', None)
        if original_fps is None:
            return f"⚠️ 跳过 {os.path.basename(input_path)}: 找不到 fps 元数据，请使用 --input_fps 强制指定"

    num_frames = data['root_pos'].shape[0]
    duration = (num_frames - 1) / original_fps

    # 2. 如果 FPS 已经一致，直接复制文件
    if abs(original_fps - target_fps) < 1e-3:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return f"➡️ 复制 {os.path.basename(input_path)} (原FPS={original_fps}，无需修改)"

    # 3. 计算新旧时间轴
    old_times = np.linspace(0, duration, num_frames)
    new_num_frames = int(duration * target_fps) + 1
    new_times = np.linspace(0, duration, new_num_frames)

    # 4. 创建新数据容器并更新元数据
    new_data = data.copy()
    
    # 💡 [核心修复 1] 修改顶层标签
    new_data['fps'] = float(target_fps)
    
    # 💡 [核心修复 2] 深度修改 meta 里的底层时间步长数组
    if 'meta' in new_data:
        if 'frame_dt_per_step' in new_data['meta']:
            new_dt = 1.0 / float(target_fps)
            # 生成长度为新帧数的时间步长数组
            new_data['meta']['frame_dt_per_step'] = np.full(new_num_frames, new_dt, dtype=np.float32)

    # 5. 执行插值
    # 包含所有的线性变量，特别注意 key_body_pos 也在里面，它会被完美插值
    linear_keys = ['root_pos', 'dof_pos', 'root_vel', 'root_vel_body', 'root_rot_vel', 'dof_vel', 'key_body_pos']
    for key in linear_keys:
        if key in data and data[key] is not None:
            new_data[key] = interpolate_linear(old_times, data[key], new_times)

    # 四元数旋转单独用 Slerp 插值
    if 'root_rot' in data:
        new_data['root_rot'] = interpolate_quaternion(old_times, data['root_rot'], new_times)

    # 6. 保存最终文件
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
        
    return f"✅ 完成 {os.path.basename(input_path)}: 帧数 {num_frames} -> {new_num_frames}"

def main():
    parser = argparse.ArgumentParser(description="终极版：批量修改 PKL 动作文件的帧率 (含底层 meta 修复)")
    parser.add_argument("--input_dir", type=str, required=True, help="输入 PKL 文件夹绝对路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 PKL 文件夹绝对路径")
    parser.add_argument("--output_fps", type=float, required=True, help="目标 FPS (例如 Isaac Lab 常用 50)")
    parser.add_argument("--input_fps", type=float, default=None, help="如果原文件缺少 fps 标签，可用此强制指定原 fps")
    args = parser.parse_args()

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 已创建输出目录: {args.output_dir}")

    # 扫描目录下所有的 pkl 文件
    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
    if not pkl_files:
        print(f"❌ 错误: 在 {args.input_dir} 目录下没有找到任何 .pkl 文件！")
        return

    print("=" * 60)
    print(f"🚀 开始批量重采样: 共发现 {len(pkl_files)} 个动作文件")
    print(f"🎯 目标帧率 (Target FPS): {args.output_fps} Hz")
    print("=" * 60)

    # 使用 tqdm 显示处理进度条
    for filename in tqdm(pkl_files, desc="处理进度", unit="file"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        # 执行处理
        result_msg = process_single_file(input_path, output_path, args.output_fps, args.input_fps)
        # 如果你想看每个文件的具体处理信息，可以取消下面这行的注释
        # tqdm.write(result_msg)

    print("\n" + "=" * 60)
    print(f"🎉 批量重采样全部完成！")
    print(f"📂 完美的新数据已存放在: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()