import os
import subprocess
from multiprocessing import Process
from numpy import random
random.seed(0)
import json

import time
import argparse
import yaml
import gc

import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def mesh_normalization(mesh, gpu) -> Meshes:
    # google
    rotation = torch.tensor([[1, 0, 0],
                             [0, 0, -1],
                             [0, 1, 0]], device=f'cuda:{gpu}', dtype=torch.float32)

    # face forward
    verts = mesh.verts_packed()
    verts = verts @ rotation
    mesh._verts_packed = verts

    # normalization
    center = (verts.max(0)[0] + verts.min(0)[0]) / 2
    scale_max = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale_max)))
    return mesh

def load_object(image_path, gpu):
    # TODO: need to be changed to load object
    # load object
    mesh = load_objs_as_meshes([image_path], device=f'cuda:{gpu}')
    return mesh_normalization(mesh, gpu)

def main_batch(gpu):
    file_path = 'objects.yaml' #'custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    objects = data['stage1_2']

    raster_settings = RasterizationSettings(
        image_size=(256, 256),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
    cameras = FoVPerspectiveCameras(device=f'cuda:{gpu}', R=R, T=T)
    lights = AmbientLights(device=f'cuda:{gpu}')

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=f'cuda:{gpu}',
            cameras=cameras,
            lights=lights
        )
    )

    for obj in objects:
        start_memory = torch.cuda.memory_allocated(f'cuda:{gpu}') / (1024 ** 2)
        obj_mesh = load_object(obj[0], gpu)
        time.sleep(5)

        load_memory = torch.cuda.memory_allocated(f'cuda:{gpu}') / (1024 ** 2)
        time.sleep(5)

        img, fragments = renderer(obj_mesh, cameras=cameras)
        time.sleep(5)

        render_memory = torch.cuda.memory_allocated(f'cuda:{gpu}') / (1024 ** 2)

        print(f'Object: {obj[-1]}, edges: {len(obj_mesh.edges_packed())}, faces: {len(obj_mesh.faces_packed())},  Memory: {load_memory - start_memory}, render: {render_memory - load_memory}')

        del obj_mesh, img, fragments
        cleanup()
        time.sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--gpu', type=int, help='GPU argument')
    args = parser.parse_args()

    if args.gpu is None:
        parser.error("You must provide both --index and --gpu arguments.")
    else:
        main_batch(gpu=args.gpu)