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
        <img src="gifs/icecream-melt-static.gif" alt="icecream static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">the ice cream is melting</h3>
        <img src="gifs/icecream-melt.gif" alt="icecream melt" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/hulk-transform-static.gif" alt="Hulk static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">the hulk is transforming</h3>
        <img src="gifs/hulk-transform.gif" alt="Hulk transforming" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/plant-bloom-static.gif" alt="Plant static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">A blooming plant slowly grows with colorful branches expanding outward</h3>
        <img src="gifs/plant-bloom.gif" alt="Plant blooming" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/candle_spell_2-static.gif" alt="candle static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a spell is cast through the purple flame</h3>
        <img src="gifs/candle_spell_2.gif" alt="candle spell" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/elephant-wings-static.gif" alt="Elephant wings static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">an elephant grows its ears into long, powerful wings, stretching wide with a graceful flap</h3>
        <img src="gifs/elephant-wings.gif" alt="Elephant wings flap" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/unicorn-rainbow-static.gif" alt="Unicorn rainbow static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a unicorn grows a colorful rainbow tail</h3>
        <img src="gifs/unicorn-rainbow.gif" alt="Unicorn rainbow tail" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/snowman-melt-static.gif" alt="Snowman static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a snowman is melting, water trickling down its sides into a pool</h3>
        <img src="gifs/snowman-melt.gif" alt="Snowman melting" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/turtle-head-static.gif" alt="Turtle static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">A turtle has its head inside its shell</h3>
        <img src="gifs/turtle-head.gif" alt="Turtle head inside" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/honey-dipper-static.gif" alt="Honey dipper static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">Honey spills from a dipper</h3>
        <img src="gifs/honey-dipper.gif" alt="Honey spilling" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/apple-bite-static.gif" alt="Apple static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">an apple with a bite taken out of it</h3>
        <img src="gifs/apple-bite.gif" alt="Apple bite" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/broccoli-grow-static.gif" alt="Broccoli static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">broccoli is growing and blooming, its green stalks stretching upward</h3>
        <img src="gifs/broccoli-grow.gif" alt="Broccoli growing" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/candle-down-melt-static.gif" alt="Candle down melt static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a candle is melting downward, wax dripping steadily</h3>
        <img src="gifs/candle-down-melt.gif" alt="Candle down melting" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/candle-melt-static.gif" alt="Candle melt static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a candle is melting, wax pooling at its base</h3>
        <img src="gifs/candle-melt.gif" alt="Candle melting" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/raccoon-fire-static.gif" alt="Raccoon fire static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a raccoon breathing fire from a weapon</h3>
        <img src="gifs/raccoon-fire.gif" alt="Raccoon fire breathing" style="max-width: 120px; height: auto;">
    </div>
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; gap: 5px; margin-bottom: 15px;">
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 18px; margin: 3px 0; width: 180px; white-space: normal;">3D Mesh</h3>
        <img src="gifs/icecream-chocholate-static.gif" alt="Ice cream chocolate static" style="max-width: 120px; height: auto;">
    </div>
    <div style="text-align: center; margin: 5px; min-width: 200px;">
        <h3 style="font-size: 16px; margin: 3px 0; width: 180px; white-space: normal;">a chocolate is poured over the ice cream and drips from its side</h3>
        <img src="gifs/icecream-chocholate.gif" alt="Ice cream chocolate dripping" style="max-width: 120px; height: auto;">
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
