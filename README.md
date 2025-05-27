# Bringing Objects to Life: training-free 4D generation from 3D objects through view consistent noise
<a href="https://three24d.github.io/three24d/"><img src="https://img.shields.io/badge/🌐%20Project-Website-blue"></a>
<a href="https://arxiv.org/abs/2412.20422"><img src="https://img.shields.io/badge/arXiv-2412.20422-b31b1b.svg?logo=arXiv"></a>
<a href="https://huggingface.co/papers/2412.20422"><img src="https://img.shields.io/badge/🤗-Hugging%20Face-orange.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
<!-- Official implementation. -->
<br>
<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/icecream-melt-static.gif" alt="icecream static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">the ice cream is melting</h3>
        <img src="gifs/arrow.gif" alt="Mario running" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">Result 4D</h3>
        <img src="gifs/icecream-melt.gif" alt="icecream melt" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;"></h3>
        <img src="gifs/candle_spell_2-static.gif" alt="candle static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a spell is cast through the purple flame</h3>
        <img src="gifs/arrow.gif" alt="Mario running" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;"></h3>
        <img src="gifs/candle_spell_2.gif" alt="candle spell" style="max-width: 120px; height: auto;">
    </div>
</div>

> <a href="https://three24d.github.io/three24d/">**Bringing Objects to Life: training-free 4D generation from 3D objects through view consistent noise**</a>
>
> <a href="https://ohadrahamim.github.io/">Ohad Rahamim</a>,
> <a href="https://github.com/Orimalca">Ori Malca</a>,
> <a href="https://chechiklab.biu.ac.il/~dvirsamuel/">Dvir Samuel</a>,
> <a href="https://chechiklab.biu.ac.il/~gal/">Gal Chechik</a>
> <br>
> Our method receives an input 3D model - like a model of your Mario figure, and a textual prompt - like ``Mario Running". Our goal is to animate the object, generating a 4D scene that reflects the described action in the prompt, yielding a 4D object of your favorite flower blooming. 
</p>

# Installation
Install the conda virtual environment:
```bash
conda create -n animate python=3.9
conda activate animate
TCNN_CUDA_ARCHITECTURES=80 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch # for A100
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

# Generate a 4D scene
Our model is trained in 2 stages, while the first can be shared between different prompts.
Every different prompt requires a new training of the second stage.
We provided two configs for every stage.

```sh
    # Stage 1
    python launch.py --config custom/configs/3to4D-stage1.yaml --train --gpu 1 exp_root_dir=outputs seed=0 data.image.object_path=\path\to\your\mesh.obj system.prompt_processor.prompt="your desiered action"
    
    # Stage 2
    python launch.py --config custom/configs/3to4D-stage2.yaml --train --gpu 1 exp_root_dir=outputs seed=0 data.image.object_path=\path\to\your\mesh.obj system.prompt_processor.obj_token_clip_idx=\the\word\location\in\the\prompt system.prompt_processor.prompt="your desiered action"
```

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="combined_gif.gif" alt="icecream static" style="max-width: 120px; height: auto;">
    </div>
</div>


## Citation
If you find this useful for your research, please cite the following:
```bibtex
@article{rahamim2024bringingobjectslife4d,
      title={Bringing Objects to Life: 4D generation from 3D objects}, 
      author={Ohad Rahamim and Ori Malca and Dvir Samuel and Gal Chechik},
      year={2024},
      eprint={2412.20422},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.20422}, 
}
```
