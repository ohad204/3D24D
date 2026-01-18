import os
import imageio
import numpy as np

def replace_red_to_white(depth):
    x = 20
    for pixel_x in range(0, 10):
        for pixel_y in range(0, 10):
            for pixel in range(125 - x, 125 + x):
                mask = np.all(depth == [pixel, pixel_x, pixel_y], axis=-1)
                depth[mask] = [255, 255, 255]

        return depth

img_dir = '/cortex/users/ohadr/experiments/object_animation/objects_frames/animate_exps'
exps = os.listdir(img_dir)
exps = [os.path.join(os.path.join(img_dir, exp)) for exp in exps if exp.endswith('inc.mp4')]
for exp in exps:
    depth = exp.replace('inc.mp4', 'inc_depth.mp4')

    reader_exp = imageio.get_reader(exp)
    reader_depth = imageio.get_reader(depth)

    # if os.path.exists(exp.replace('.mp4', '')):
    #     continue

    os.makedirs(exp.replace('.mp4', ''), exist_ok=True)

    for idx, (img, depth) in enumerate(zip(reader_exp, reader_depth)):
        out_path = os.path.join(exp.replace('.mp4', ''), 'frame_%04d.png' % idx)
        img = img[25:-25, 25:-25]
        width, height, _ = img.shape
        depth = depth[25:-25, 25:-25]
        # depth = replace_red_to_white(depth)
        # merged_image = np.concatenate((img[:, :width // 2], depth[:, width // 2:]), 1)

        merged_image = img
        merged_image[:110, 103:] = depth[:110, 103:]

        imageio.imwrite(out_path, merged_image)