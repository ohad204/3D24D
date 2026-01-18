import numpy as np
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
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

import os
import torch
import torch.nn.functional as F
import yaml
import imageio
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device


class MeshRenderer:
    def __init__(self, image_path):
        self.device = torch.device(f"cuda:3")

        self.image_path = image_path
        self.load_object()

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        self.raster_settings = RasterizationSettings(
            image_size=256,
            # image_size=(320, 512),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Initialize a camera.
        R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        lights = AmbientLights(device=self.device)
        # lights = PointLights(device=self.device, location=light_pos)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )

    def render(self, cam_elev, cam_azim):
        if azimuth < 180:
            light_location = torch.tensor([1.5, 0.0, 1.5], device=self.device).view(1, 1, 3)
        else:
            light_location = torch.tensor([-1.5, 0.0, -1.5], device=self.device).view(1, 1, 3)

        lights = DirectionalLights(
            device=self.device,
            direction=light_location,
            # ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.5, 0.5, 0.5]],
            # specular_color=[[0.5, 1.0, 0.5]]
        )

        R, T = look_at_view_transform(dist=2.5, elev=cam_elev,
                                      azim=cam_azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        img, fragments = self.renderer(self.obj, cameras=cameras, lights=lights)

        return img

    def prepared_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth[depth == -1] = 0
        return depth

    def phong_normal_shading(self, fragments) -> torch.Tensor:
        faces = self.obj.faces_packed()  # (F, 3)
        vertex_normals = self.obj.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        pixel_normals = F.normalize(pixel_normals, p=2, dim=-1)
        return pixel_normals[..., 0, :]


    def mesh_normalization(self, mesh) -> Meshes:
        # google
        rotation = torch.tensor([[1, 0, 0],
                                 [0, 0, -1],
                                 [0, 1, 0]], device=self.device, dtype=torch.float32)

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


    def load_object(self):
        # TODO: need to be changed to load object
        # load object
        assert os.path.exists(
            self.image_path
        ), f"Could not find image {self.image_path}!"
        mesh = load_objs_as_meshes([self.image_path], device=self.device)
        self.obj = self.mesh_normalization(mesh)

file_path = 'objects.yaml'
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

objects = data['stage2']

for object in objects:
    renderer = MeshRenderer(object[0])
    output_dir = f'/home/lab/rahamiob/ohadr/experiments/object_animation/seeds/images_light/{object[3]}'
    os.makedirs(output_dir, exist_ok=True)

    for azimuth in [0, 90, 180, 270]:
        img = renderer.render(0, azimuth)
        img = img[0, ..., :3].cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)

        output_path = os.path.join(output_dir, f'{azimuth}.png')
        imageio.imwrite(output_path, img)