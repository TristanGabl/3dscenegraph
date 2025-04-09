import argparse
import os
import shutil
import numpy as np
import cv2
import json
import trimesh


def process_scans(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    
    scan_name = input_folder.split('/')[-2]
    output_folder = os.path.join(output_folder, input_folder.split('/')[-2])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_numbers = [scan_file.split('.')[0].split('-')[1] for scan_file in os.listdir(input_folder) if scan_file.endswith('.pose.txt')]
    frames_numbers.sort()

    # Randomly drop 80% of the frames
    np.random.seed(42)  # For reproducibility
    total_frames = len(frames_numbers)
    frames_to_keep = int(total_frames * 0.1)
    frames_numbers = np.random.choice(frames_numbers, frames_to_keep, replace=False)

    info_file = os.path.join(input_folder, "_info.txt") # same for all
    # Read the info file to extract calibration data
    with open(info_file, 'r') as f:
        lines = f.readlines()

    # Extract m_calibrationDepthIntrinsic and m_calibrationColorIntrinsic
    depth_intrinsic_line = next(line for line in lines if line.startswith("m_calibrationDepthIntrinsic"))
    color_intrinsic_line = next(line for line in lines if line.startswith("m_calibrationColorIntrinsic"))
    color_width_line = next(line for line in lines if line.startswith("m_colorWidth"))
    color_height_line = next(line for line in lines if line.startswith("m_colorHeight"))
    depth_width_line = next(line for line in lines if line.startswith("m_depthWidth"))
    depth_height_line = next(line for line in lines if line.startswith("m_depthHeight"))
    depth_shift_line = next(line for line in lines if line.startswith("m_depthShift"))
    
    depth_intrinsic = np.array(list(map(float, depth_intrinsic_line.split('=')[1].strip().split()))).reshape(4, 4)
    color_intrinsic = np.array(list(map(float, color_intrinsic_line.split('=')[1].strip().split()))).reshape(4, 4)
    color_width = int(color_width_line.split('=')[1].strip())
    color_height = int(color_height_line.split('=')[1].strip())
    depth_width = int(depth_width_line.split('=')[1].strip())
    depth_height = int(depth_height_line.split('=')[1].strip())
    depth_shift = float(depth_shift_line.split('=')[1].strip())


    # save ply file 
    

    ply_file = input_folder.split('SensReader_out')[0] + f"{scan_name}_vh_clean_2.ply"
    if not os.path.exists(ply_file):
        print(f"PLY file '{ply_file}' does not exist?")
        return

    # move the ply file to the output folder
    # Load the PLY file using trimesh
    mesh = trimesh.load(ply_file)
    # Save the mesh as an OBJ file in the output folder
    obj_file_path = os.path.join(output_folder, f"export_refined.obj")
    mesh.visual.vertex_normals = None
    mesh.export(obj_file_path)

    for frame_number in frames_numbers:
        image_file = os.path.join(input_folder, f"frame-{frame_number}.color.jpg")
        depth_file = os.path.join(input_folder, f"frame-{frame_number}.depth.pgm")
        pose_file = os.path.join(input_folder, f"frame-{frame_number}.pose.txt")

        if not os.path.exists(image_file) or not os.path.exists(depth_file) or not os.path.exists(pose_file):
            print(f"Missing files for frame {frame_number}. Skipping...")
            continue
        
        # just rename image_file to image_{frame_number}.jpg
        new_image_file = os.path.join(output_folder, f"frame_{frame_number}.jpg")
        shutil.copy(image_file, new_image_file)

        # we load depth and safe as png
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(output_folder, f"depth_{frame_number}.png"), depth)

        # we load pose 
        with open(pose_file, 'r') as f:
            pose = np.array([list(map(float, line.split())) for line in f.readlines()])



        # image_info = json.load(open("scans/three_books_scan/frame_00004.json", 'r'))
        # projectionMatrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
        # cameraPoseARFrame = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
        # compute the projection matrix

        # Calculate MVP matrix (projection * view)
        
        # Save the MVP matrix into a JSON file
        json_file = os.path.join(output_folder, f"frame_{frame_number}.json")
        with open(json_file, 'w') as f:
            json.dump({
                'calibrationColorIntrinsic': color_intrinsic.tolist(),
                'calibrationDepthIntrinsic': depth_intrinsic.tolist(),
                'Pose': pose.tolist(),
                'depthShift': depth_shift,
                'depthWidth': depth_width,
                'depthHeight': depth_height,
                'colorWidth': color_width,
                'colorHeight': color_height
            }, f, indent=4)
        
                
## 
# pose = np.array(image_info['cameraPoseARFrame']).reshape((4, 4))
# projection_matrix = np.array(image_info['projectionMatrix']).reshape((4, 4))
# view_matrix = np.linalg.inv(pose)
# mvp = np.dot(projection_matrix, view_matrix)

# def project_points_to_image(self, p_in, mvp, image_width, image_height):
#     p0 = np.concatenate([p_in, np.ones([p_in.shape[0], 1])], axis=1)
#     e0 = np.dot(p0, mvp.T)
#     pos_z = e0[:, 2]
#     e0 = (e0.T / e0[:, 3]).T
#     pos_x = e0[:, 0]
#     pos_y = e0[:, 1]
#     projections = np.zeros([p_in.shape[0], 3])
#     projections[:, 0] = (0.5 + (pos_x) * 0.5) * image_width
#     projections[:, 1] = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
#     projections[:, 2] = pos_z  # Store the z coordinate
#     return projections
    





    print(f"Processing complete. Output saved to '{output_folder}'.")

def main():
    parser = argparse.ArgumentParser(description="Process ScanNet dataset.")
    parser.add_argument('--input', default="ScanNet/scans/scene0134_02/SensReader_out", help="Path to the input scan folder (default: 'ScanNet/scans/scene0134_02/SensReader_out').")
    parser.add_argument('--output', default="scans/", help="Path to the output scan folder (default: 'scans/').")
    parser.add_argument('--drop', type=float, default=0.9, help="Scaling factor for dropping frames (default: 0.9).")
    args = parser.parse_args()

    process_scans(args.input, args.output)

if __name__ == "__main__":
    main()