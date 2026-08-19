from  qm9.visualizer import *
from configs.datasets_config import get_dataset_info
from rdkit import Chem
import argparse
from qm9.rdkit_functions import BasicMolecularMetrics, build_molecule
import os
from PIL import Image, ImageDraw, ImageFont
def visualize(file, dataset_info, spheres_3d=False):
    positions, one_hot, charges = load_molecule_xyz(file, dataset_info)
    
    # Filter positions with absolute values less than or equal to 20
    mask = (torch.abs(positions) <= 20).all(dim=1)
    filtered_positions = positions[mask]
    filtered_atom_type = torch.argmax(one_hot, dim=1).numpy()[mask]
    
    # Recalculate distances after filtering
    dists = torch.cdist(filtered_positions.unsqueeze(0), filtered_positions.unsqueeze(0)).squeeze(0)
    dists = dists[dists > 0]
    print(file)
    print("Average distance between atoms", dists.mean().item())
    print(f"Original number of atoms: {positions.shape[0]}")
    print(f"Filtered number of atoms: {filtered_positions.shape[0]}")
    
    plot_data3d(filtered_positions, filtered_atom_type, dataset_info=dataset_info, 
                save_path=file[:-4] + '.png', spheres_3d=spheres_3d)
    
    # positions, one_hot, charges = load_molecule_xyz(file, dataset_info)
    atom_type = torch.argmax(one_hot, dim=1).numpy()

    dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
    dists = dists[dists > 0]

    # print("Average distance between atoms", dists.mean().item())
    # plot_data3d(positions, atom_type, dataset_info=dataset_info, save_path=file[:-4] + '.png',
    #             spheres_3d=spheres_3d)


    sdf_writer = Chem.SDWriter(file[:-4] + '.sdf')
    mol = build_molecule(*(positions, atom_type), dataset_info)
    if mol is not None:  # Ensure the molecule is not None before writing
        # Optionally add dataset information as properties to the molecules
        for key, value in dataset_info.items():
            mol.SetProp(key, str(value))
        
        # Step 3: Write the molecule to the SDF file
        sdf_writer.write(mol)
    
    # Step 4: Close the SDF writer
    sdf_writer.close()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a molecular animation GIF from XYZ files in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing XYZ files (chain_*.txt)")
    args = parser.parse_args()
    dataset_info = get_dataset_info('qm9', False)


    # 假设 args.input_dir 是用户提供的路径
    if os.path.isdir(args.input_dir):
        # 遍历目录中的所有文件
        num_frames = 100
        # background = Image.new("RGBA", (443, 443), (255, 255, 255))  # 白色背景
        for idx, file in enumerate(sorted(os.listdir(args.input_dir))):
            file_path = os.path.join(args.input_dir, file)
            if file[-3:] == 'txt':
                if os.path.isfile(file_path):  # 确保是文件而非子目录
                    visualize(file_path, dataset_info=dataset_info, spheres_3d=True)
            
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except IOError:
                font = ImageFont.load_default()        
            # 保存帧
            frame_path = f"{args.input_dir}/{file[:-4]}"+ '.png'
            img = Image.open(frame_path)
            draw = ImageDraw.Draw(img)
            print(img.size)
            draw.text(
                (img.width - 200, img.height - 50),  # 右下角位置
                f"step={frame_path[-7:-4]}",  # 步骤文字
                fill=(255, 255, 255), 
                font=font
            )
            img.save(frame_path)

            # 设置帧的持续时间
            # durations = [30] * num_frames
            # for i in range(num_frames - 20, num_frames):  
            #     durations[i] = 100  
            # for i in range(num_frames - 10, num_frames):  
            #     durations[i] = 200  
            # durations[-1] = 100000
            
            
        # frames = [Image.open(f"{args.input_dir}/chain_{idx:03d}.png").convert("RGBA") for idx in range(0, num_frames)]
        # background = Image.new("RGBA", frames[0].size, (255, 255, 255))  # 白色背景

        # converted_frames = []
        # for frame in frames:
        #     bg = background.copy()
        #     bg.paste(frame, (0, 0), frame)
        #     converted_frames.append(bg.convert("RGB"))
        frames = [Image.open(f"{args.input_dir}/chain_{idx:03d}.png") for idx in range(0, num_frames)]
        frames[0].save('./animation.gif', save_all=True, append_images=frames[1:], optimize=False, duration=50, loop=1, disposal=2)
        print(f"GIF saved to ./animation.gif")
            
    else:
        # 如果不是目录，直接调用 visualize
        visualize(args.input_dir, dataset_info=dataset_info, spheres_3d=True)
