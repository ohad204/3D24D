import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

from pytorch3d.io import load_objs_as_meshes, IO, load_obj
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    BlendParams,
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

LATENT_SHAPE = 32

@dataclass
class SingleObjectDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    rotate: int = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False

    lights: bool = False

    rays_d_normalize: bool = True

    sphere_level: int = 6
    sphere_scale: float = 1.7
    noise_reduction: int = 1

    dropout_rate:float = - 1.0
    first_faces:bool = False

    global_noise_scale: float = 1.39


class SingleObjectDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleObjectDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.load_object()
        self.prev_height = self.height

        self.sphere_mesh = ico_sphere(self.cfg.sphere_level, self.rank)
        self.sphere_mesh.scale_verts_(self.cfg.sphere_scale)
        verts = self.sphere_mesh.verts_packed()  # Shape: (V, 3)
        textures = TexturesVertex(verts_features=torch.zeros(1, verts.shape[0], 3, device=self.rank))
        self.sphere_mesh.textures = textures
        self.latent = torch.randn((4, 16, self.sphere_mesh.faces_packed().shape[0]), device=self.rank)

        self.noise_reduction = 4
        noise_size = int(self.obj._faces_packed.shape[0] / self.noise_reduction) + 1
        self.faces_noise = torch.randn((4, noise_size), device=self.rank)

        # Store latent noise per vertex of the single sphere mesh
        num_verts_sphere = self.sphere_mesh.verts_packed().shape[0]
        # Latent shape: (batch_size, channels, height, width) -> (1, 4, 16, 32, 32)
        # We need 4 * 16 = 64 values per vertex
        self.vertex_latent_noise = torch.randn(
            (4 * 16, num_verts_sphere), device=self.rank
        ) # Shape: (64, num_verts_sphere)
        self.vertex_latent_noise = self.vertex_latent_noise * self.cfg.global_noise_scale

        # Get vertex noise values per face for the single sphere mesh
        faces_sphere_single = self.sphere_mesh.faces_packed() # (F_s, 3)
        # vertex_noise_values_per_face_single shape: (64, F_s, 3)
        self.vertex_noise_values_per_face_single = self.vertex_latent_noise[:, faces_sphere_single]
        # Permute to (F_s, 3, 64) for interpolation - THIS TENSOR WILL BE PASSED WITHOUT BATCH DIM
        self.sphere_vertex_attributes_for_interpolation = self.vertex_noise_values_per_face_single.permute(1, 2, 0)

    def setup_2(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleObjectDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.random_camera
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.load_object()
        self.prev_height = self.height

        self.sphere_mesh = ico_sphere(self.cfg.sphere_level, self.rank)
        self.sphere_mesh.scale_verts_(self.cfg.sphere_scale)
        verts = self.sphere_mesh.verts_packed()  # Shape: (V, 3)
        textures = TexturesVertex(verts_features=torch.zeros(1, verts.shape[0], 3, device=self.rank))
        self.sphere_mesh.textures = textures
        self.latent = torch.randn((4, 16, self.sphere_mesh.faces_packed().shape[0]), device=self.rank)

        self.noise_reduction = 4
        noise_size = int(self.obj._faces_packed.shape[0] / self.noise_reduction) + 1
        self.faces_noise = torch.randn((4, noise_size), device=self.rank)

        # Store latent noise per vertex of the single sphere mesh
        num_verts_sphere = self.sphere_mesh.verts_packed().shape[0]
        # Latent shape: (batch_size, channels, height, width) -> (1, 4, 16, 32, 32)
        # We need 4 * 16 = 64 values per vertex
        self.vertex_latent_noise = torch.randn(
            (4 * 16, num_verts_sphere), device=self.rank
        ) # Shape: (64, num_verts_sphere)
        self.vertex_latent_noise = self.vertex_latent_noise * self.cfg.global_noise_scale

        # Get vertex noise values per face for the single sphere mesh
        faces_sphere_single = self.sphere_mesh.faces_packed() # (F_s, 3)
        # vertex_noise_values_per_face_single shape: (64, F_s, 3)
        self.vertex_noise_values_per_face_single = self.vertex_latent_noise[:, faces_sphere_single]
        # Permute to (F_s, 3, 64) for interpolation - THIS TENSOR WILL BE PASSED WITHOUT BATCH DIM
        self.sphere_vertex_attributes_for_interpolation = self.vertex_noise_values_per_face_single.permute(1, 2, 0)

    def prepared_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth[depth == -1] = 0
        return depth

    def phong_normal_shading(self, fragments) -> torch.Tensor:
        faces = self.obj[0].faces_packed()  # (F, 3)
        vertex_normals = self.obj[0].verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        pixel_normals = F.normalize(pixel_normals, p=2, dim=-1)
        return pixel_normals[..., 0, :]

    def mesh_normalization(self, mesh, rotate=1, scale=1.0) -> Meshes:
        # google
        if rotate == 1:
            rotation = torch.tensor([[1, 0,  0],
                                     [0, 0, -1],
                                     [0, 1,  0]], device=self.rank, dtype=torch.float32)
        # panda 2
        if rotate == 2:
            rotation = torch.tensor([[0,  0,  1],
                                           [0,  1, 0],
                                           [-1, 0, 0]], device=self.rank, dtype=torch.float32)
        # panda 3
        if rotate == 3:
            rotation = torch.tensor([[0,  0,  -1],
                                           [0,  1, 0],
                                           [1,  0, 0]], device=self.rank, dtype=torch.float32)

        if rotate == 4:
            rotation = torch.tensor([[1, 0,  0],
                                           [0, -1, 0],
                                           [0, 0,  1]], device=self.rank, dtype=torch.float32)
        if rotate == 5:
            rotation = torch.tensor([[1, 0,  0],
                                           [0, -1, 0],
                                           [0, 0,  -1]], device=self.rank, dtype=torch.float32)
        if rotate == 6:
            rotation = torch.tensor([[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]], device=self.rank, dtype=torch.float32)


        # face forward
        verts = mesh.verts_packed()
        if rotate > 0:
            verts = verts @ rotation
            mesh._verts_packed = verts

        # normalization
        center = (verts.max(0)[0] + verts.min(0)[0]) / 2
        scale_max = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((scale / float(scale_max)))
        return mesh

    def load_object(self):
        # TODO: need to be changed to load object
        # load object
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"
        if self.cfg.image_path.endswith(".obj"):
            mesh = load_objs_as_meshes([self.cfg.image_path], device=self.rank)
        if self.cfg.image_path.endswith(".glb"):
            from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
            io = IO()
            io.register_meshes_format(MeshGlbFormat())
            mesh = io.load_mesh(self.cfg.image_path, device=f'cuda:{self.rank}')
        self.obj = self.mesh_normalization(mesh, rotate=self.cfg.rotate)

    def get_all_objects(self):
        return self.obj

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_object()


class SingleObjectIterableDataset(IterableDataset, SingleObjectDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        if split == 'aaa':
            self.setup_2(cfg, split)
        else:
            self.setup(cfg, split)
        self.device = torch.device(f"cuda:{self.rank}")

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        self.raster_settings = RasterizationSettings(
            image_size=self.height,
            # image_size=(320, 512),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.raster_latent_settings = RasterizationSettings(
            image_size=(LATENT_SHAPE, LATENT_SHAPE),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Initialize a camera.
        R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        lights = AmbientLights(device=self.device)

        background_color = (128 / 255.0, 128 / 255.0, 128 / 255.0)
        blend_params = BlendParams(background_color=background_color)

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
                lights=lights,
                # blend_params=blend_params
            )
        )

        self.renderer_latent = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_latent_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                # blend_params=blend_params
            )
        )

    def sample_interpolated_vertex_noise(self, fragments_latents: Any) -> torch.Tensor:
        """
        Samples and interpolates the vertex_latent_noise based on batched sphere rendering fragments.

        Args:
            fragments_latents: The fragments output from rasterizing the sphere mesh batch.
                             Contains pix_to_face (N, H, W, K) and bary_coords (N, H, W, K, 3).

        Returns:
            A tensor of shape (N, 4, 16, H, W) containing the interpolated noise.
            Pixels not hitting the sphere are set to NaN. H=32, W=32.
        """
        N, H, W, K = fragments_latents.pix_to_face.shape
        # K is expected to be 1 from raster_latent_settings

        # Pass the pre-computed vertex attributes for a single sphere mesh (shape F, V, D)
        # interpolate_face_attributes will broadcast these over the batch of fragments (N, H, W, K, ...)
        interpolated_noise = interpolate_face_attributes(
            fragments_latents.pix_to_face, # (N, H, W, K)
            fragments_latents.bary_coords, # (N, H, W, K, 3)
            self.sphere_vertex_attributes_for_interpolation # Pass shape (F_s, 3, 64)
        ) # Expected output shape (N, H, W, K, D) -> (N, 32, 32, 1, 64) if K=1

        # Select K=1 dimension
        interpolated_noise = interpolated_noise[..., 0, :] # (N, 32, 32, 64)

        # Apply mask: Pixels with pix_to_face == -1 should be NaN
        # Mask shape (N, H, W, K) -> take K=0 -> (N, H, W)
        mask = fragments_latents.pix_to_face[..., 0] >= 0 # Shape (N, 32, 32)
        # Expand mask to match noise shape (N, 32, 32, 64)
        mask = mask.unsqueeze(-1).expand_as(interpolated_noise)
        interpolated_noise[~mask] = float("nan")

        if self.cfg.dropout_rate > 0.0 and self.cfg.dropout_rate < 1.0:
            dropout_random_pattern = torch.rand_like(interpolated_noise)
            dropout_mask_replace = dropout_random_pattern < self.cfg.dropout_rate
            new_random_noise = torch.randn_like(interpolated_noise)
            interpolated_noise[dropout_mask_replace] = new_random_noise[dropout_mask_replace]

        # Reshape to (N, 4, 16, 32, 32)
        # (N, 32, 32, 64) -> view(N, 32, 32, 4, 16) -> permute(0, 3, 4, 1, 2)
        object_noise = interpolated_noise.view(N, LATENT_SHAPE, LATENT_SHAPE, 4, 16).permute(0, 3, 4, 1, 2)

        return object_noise[0]

    def collate(self, batch) -> Dict[str, Any]:

        if batch['camera_distances'].size() == torch.Size([]): # in validation
            cam_dist = batch['camera_distances']
            cam_elev = batch['elevation']
            cam_azim = batch['azimuth']
            light_pos = batch['light_positions']
        # elif batch['camera_distances'].shape[0] > 1:
        #     cam_dist = batch['camera_distances'][0]
        #     cam_elev = batch['elevation'][0]
        #     cam_azim = batch['azimuth'][0]
        #     light_pos = batch['light_positions'][0].unsqueeze(0)
        else:
            cam_dist = batch['camera_distances']
            cam_elev = batch['elevation']
            cam_azim = batch['azimuth']
            light_pos = batch['light_positions']

        if self.cfg.lights:
            if cam_azim < 180:
                light_location = torch.tensor([1.5, 0.0, 1.5], device=self.device).view(1, 1, 3)
            else:
                light_location = torch.tensor([-1.5, 0.0, -1.5], device=self.device).view(1, 1, 3)

            lights = DirectionalLights(
                device=self.device,
                direction=light_location,
                diffuse_color=[[0.5, 0.5, 0.5]],
            )
        else:
            lights = AmbientLights(device=self.device)

        # Initialize a camera.
        R, T = look_at_view_transform(dist=cam_dist, elev=cam_elev,
                                      azim=cam_azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        img, fragments = self.renderer(self.obj, cameras=cameras, lights=lights)
        _, fragments_latents = self.renderer_latent(self.sphere_mesh, cameras=cameras)
        if self.cfg.first_faces:
            object_noise = self.find_faces_mesh(fragments, 4)
        else:
            object_noise = self.sample_interpolated_vertex_noise(fragments_latents)
        # object_noise[:, 0] = first_noise
        normals = self.phong_normal_shading(fragments)
        depth = self.prepared_depth(fragments.zbuf)

        batch = {
            "rgb": img[..., :3],
            "depth": depth,
            "normals": normals,
            "mask": img[..., -1].unsqueeze(3),
            "object_noise": object_noise,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def collate_2(self, name) -> Dict[str, Any]:

        latents = []
        renders = []
        elev = 0
        cam_dist = 2.5
        for azi in range(360):
            R, T = look_at_view_transform(dist=cam_dist, elev=elev,
                                          azim=azi)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            img, fragments = self.renderer(self.obj, cameras=cameras)
            _, fragments_latents = self.renderer_latent(self.sphere_mesh, cameras=cameras)
            object_noise = self.sample_interpolated_vertex_noise(fragments_latents)
            latents.append(object_noise)
            renders.append(img)


        torch.save(renders, f'/home/lab/rahamiob/Archive/3to4d/renders_{name}.pt')
        torch.save(latents, f'/home/lab/rahamiob/Archive/3to4d/latents_{name}.pt')

    def find_faces_mesh(self, fragments, noise_reduction=4):
        face_indices = fragments.pix_to_face / noise_reduction
        downsampled_faces = F.interpolate(face_indices.permute(3, 0, 1, 2).float(), size=(32, 32), mode='nearest-exact')
        downsampled_faces = downsampled_faces[0, 0]
        downsampled_idx = F.interpolate(fragments.pix_to_face.permute(3, 0, 1, 2).float(), size=(32, 32),
                                        mode='nearest-exact')
        downsampled_idx = downsampled_idx[0, 0]
        object_noise_ = downsampled_faces.repeat(4, 1, 1)
        object_noise_[:, downsampled_idx < 0] = float("nan")
        object_noise_[:, downsampled_idx >= 0] = self.faces_noise[:, downsampled_faces[downsampled_faces >= 0].long()]

        object_noise = torch.full((4, 16, 32, 32), float('nan'))
        object_noise[:, 0, :, :] = object_noise_

        return object_noise

    def find_faces(self, fragments, noise_reduction=1):
        face_indices = fragments.pix_to_face / noise_reduction
        downsampled_faces = F.interpolate(face_indices.permute(3, 0, 1, 2).float(), size=(32, 32), mode='nearest-exact')
        downsampled_faces = downsampled_faces[0, 0]
        downsampled_idx = F.interpolate(fragments.pix_to_face.permute(3, 0, 1, 2).float(), size=(32, 32),
                                        mode='nearest-exact')
        downsampled_idx = downsampled_idx[0, 0]
        object_noise = downsampled_faces.repeat(4, 16, 1, 1)
        object_noise[:, :, downsampled_idx < 0] = float("nan")
        object_noise[:, :, downsampled_idx >= 0] = self.latent[:, :, downsampled_faces[downsampled_faces >= 0].long()]
        return object_noise

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleObjectDataset(Dataset, SingleObjectDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]


if __name__ == '__main__':
    import yaml
    file_path = '/home/lab/rahamiob/ohadr/projects/object_animation/custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    objects = data['stage2-objaverse-test']

    cfg = SingleObjectDataModuleConfig()
    for object in objects:
        cfg.image_path = object[0]
        cfg.rotate = int(object[1])
        dataset = SingleObjectIterableDataset(cfg, 'aaa')
        dataset.collate_2(object[3])