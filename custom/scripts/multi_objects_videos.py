import os
from shutil import copyfile

root_dir = '/home/lab/rahamiob/ohadr/projects/object_animation/output-validation-60/animate124-stage2'
output_dir = '/home/lab/rahamiob/ohadr/projects/object_animation/output-validation-60/videos'
experiments = os.listdir(root_dir)

for exp in experiments:
    video_path = os.path.join(root_dir, exp, 'save/it0-test.mp4')
    static_path = os.path.join(root_dir, exp, 'save/it0-test_static.mp4')
    prompt = exp.split('@')[0]

    if os.path.exists(video_path):
        copyfile(video_path, os.path.join(output_dir, f'{prompt}.mp4'))
        copyfile(static_path, os.path.join(output_dir, f'{prompt}-static.mp4'))