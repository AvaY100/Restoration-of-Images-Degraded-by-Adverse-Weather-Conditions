import os
import random
import shutil

def main():
    # Directories and file path setup
    gt_dir = '/home/ec2-user/TransWeather/data_onetenth/train/gt'
    input_dir = '/home/ec2-user/TransWeather/data_onetenth/train/input'
    depth_input_file = '/home/ec2-user/TransWeather/data_onetenth/train/depth_input.txt'

    # Get all picture filenames in the gt directory
    gt_pictures = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly select 10% of these pictures to keep
    num_to_keep = max(1, len(gt_pictures) // 10)
    keep_pictures = set(random.sample(gt_pictures, num_to_keep))

    # Delete all other pictures from gt and input directories
    for picture in gt_pictures:
        gt_path = os.path.join(gt_dir, picture)
        input_path = os.path.join(input_dir, picture)
        if picture not in keep_pictures:
            os.remove(gt_path)
            os.remove(input_path)

    # Rewrite the depth_input.txt file
    with open(depth_input_file, 'w') as file:
        for picture in keep_pictures:
            file.write(f'./input/{picture}\n')

if __name__ == '__main__':
    main()
