import os

import numpy as np

# Root directory
root_dir = '/home/lab/rahamiob/www/object_animation/video_prev'

# Main title string
main_title = 'Visualize timesteps rendering results'

# Technical details string
technical_details = 'DynamiCrafter SDS'

# Absolute path to the directory where the HTML file should be saved
output_dir = '/home/lab/rahamiob/www/object_animation'

# Absolute path to the output HTML file
output_file_path = os.path.join(output_dir, 'video_prev.html')

# Get the list of subdirectories (prompts)
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

with open(output_file_path, 'w') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<html lang="en">\n')
    f.write('<head>\n')
    f.write('    <meta charset="UTF-8">\n')
    f.write(f'    <title>{main_title}</title>\n')
    f.write('    <style>\n')
    f.write('        body { text-align: center; font-family: Arial, sans-serif; }\n')
    f.write('        h1, h2 { margin-bottom: 20px; }\n')
    f.write('        .prompt { display: none; }\n')
    f.write('        .active { display: block; }\n')
    f.write('        button { font-size: 30px; cursor: pointer; padding: 10px 20px; margin: 20px; }\n')
    f.write('        img { display: block; margin: 0 auto 20px; }\n')
    f.write('        .steps { display: flex; flex-direction: column; align-items: center; }\n')
    f.write('        .technical-details { font-size: 24px; margin-bottom: 20px; } \n')
    f.write('    </style>\n')
    f.write('    <script>\n')
    f.write('        let currentIndex = 0;\n')
    f.write('        window.onload = function() {\n')
    f.write('            const prompts = document.getElementsByClassName("prompt");\n')
    f.write('            prompts[currentIndex].classList.add("active");\n')
    f.write('        }\n')
    f.write('        function showPrompt(index) {\n')
    f.write('            const prompts = document.getElementsByClassName("prompt");\n')
    f.write('            prompts[currentIndex].classList.remove("active");\n')
    f.write('            currentIndex = index;\n')
    f.write('            prompts[currentIndex].classList.add("active");\n')
    f.write('        }\n')
    f.write('        function nextPrompt() {\n')
    f.write('            const prompts = document.getElementsByClassName("prompt");\n')
    f.write('            showPrompt((currentIndex + 1) % prompts.length);\n')
    f.write('        }\n')
    f.write('        function prevPrompt() {\n')
    f.write('            const prompts = document.getElementsByClassName("prompt");\n')
    f.write('            showPrompt((currentIndex - 1 + prompts.length) % prompts.length);\n')
    f.write('        }\n')
    f.write('    </script>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write(f'    <h1>{main_title}</h1>\n')
    f.write(f'    <p class="technical-details">{technical_details}</p>\n')
    f.write(f'    <div class="prompt">\n')

    videos_names = os.listdir(root_dir)
    videos_path = [os.path.join(root_dir, video_name) for video_name in videos_names]
    times = np.array([int(p.split('_')[-3]) for p in videos_path])
    times -= times.min()
    times = times.tolist()
    # videos_path = videos_path[::20]
    f.write('<table>\n')
    for video, time in zip(videos_path, times):
        gif_video = os.path.relpath(video, output_dir)
        t_noise = gif_video.split('_')[4].split('.')[0]
        line = f'<td>'
        line += f'<br> time from train start: {time}, t_noise: {t_noise}'
        line += f'</td>\n'

        f.write('<tr>')
        f.write(line)
        f.write(f'        <td><img src="{gif_video}" alt="stable diffusion Image" width="256"></td>\n')

        f.write('</tr>')
    f.write('</table>\n')

    f.write('</body>\n')
    f.write('</html>\n')

