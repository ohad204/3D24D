from dataclasses import dataclass, field
import inspect
from tqdm import tqdm
import numpy as np
import json

import torch
from torch import Tensor
import safetensors.torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline#, LTXVideoTransformer3DModel
from ltx_video.transformer_ltx import LTXVideoTransformer3DModel
from diffusers.image_processor import VaeImageProcessor
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.video_autoencoder import (
    Downsample3D,
    VideoAutoencoder,
)
from diffusers import AutoencoderKL

from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    vae_decode,
    vae_encode,
)
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.conditioning_method import ConditioningMethod


def load_vae(vae_dir):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)


def load_unet(unet_dir):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)

@threestudio.register("animate124-ltxv-guidance")
class LTXVGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = None
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True
        max_steps: int = 10000

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
        loss_attention_masking: bool = False

        low_ram_vae: int = -1

        timestep_strategy: str = "uniform"


    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        # self.weights_dtype = (
        #     torch.float16 if self.cfg.half_precision_weights else torch.float32
        # )
        self.weights_dtype = torch.float32

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path,
        #     **pipe_kwargs,
        # ).to(self.device)

        # Extra modules
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path, subfolder="tokenizer",
        #     torch_dtype=self.weights_dtype
        # )
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path, subfolder="text_encoder",
        #     torch_dtype=self.weights_dtype
        # )
        # self.text_encoder = self.text_encoder.to(self.device)

        # self.pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", cache_dir='/data/ohadr')

        from pathlib import Path
        ckpt_dir = Path(self.cfg.pretrained_model_name_or_path)
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"

        # Load models
        vae = load_vae(vae_dir)
        unet = load_unet(unet_dir)
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        from transformers import T5EncoderModel, T5Tokenizer
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )
        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }
        from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
        self.pipe = LTXVideoPipeline(**submodel_dict)
        self.pipe = self.pipe.to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.transformer.eval()
        self.scheduler = self.pipe.scheduler
        self.patchifier = SymmetricPatchifier(patch_size=1)

        # self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
        #     self.vae
        # )
        self.image_processor = VaeImageProcessor()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.max_values = np.linspace(self.max_step, int(self.max_step / 2), self.cfg.max_steps).astype(int)
        self.uc_img_emb = None
        self.fs = torch.tensor([3], dtype=torch.long, device=self.device)

        # self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
        #     self.device
        # )

        # Extra for latents
        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        threestudio.info(f"Loaded Stable Diffusion!")

    @staticmethod
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
            scheduler,
            num_inference_steps: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = None,
            timesteps: Optional[List[int]] = None,
            **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                    Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                    timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                    must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    @staticmethod
    def get_vae_size_scale_factor(vae: AutoencoderKL) -> float:
        if isinstance(vae, CausalVideoAutoencoder):
            spatial = vae.spatial_downscale_factor
            temporal = vae.temporal_downscale_factor
        else:
            down_blocks = len(
                [
                    block
                    for block in vae.encoder.down_blocks
                    if isinstance(block.downsample, Downsample3D)
                ]
            )
            spatial = vae.config.patch_size * 2 ** down_blocks
            temporal = (
                vae.config.patch_size_t * 2 ** down_blocks
                if isinstance(vae, VideoAutoencoder)
                else 1
            )

        return (temporal, spatial, spatial)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        scale_grid = (torch.tensor([[0.3200], [0.3200]], device=latents.device), 32, 32)
        indices_grid = self.patchifier.get_grid(
            orig_num_frames=3,
            orig_height=8,
            orig_width=8,
            batch_size=2,
            scale_grid=scale_grid,
            device=latents.device,
        )

        return self.unet(
                        latents.to(self.unet.dtype),
                        indices_grid,
                        encoder_hidden_states=encoder_hidden_states[0].to(self.unet.dtype),
                        encoder_attention_mask=encoder_hidden_states[1],
                        timestep=t.repeat(latents.shape[1], 1).T.to(self.weights_dtype),
                        return_dict=False,
                    )[0]

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 N 320 576"], normalize: bool = True
    ) -> Float[Tensor, "B 4 40 72"]:
        # breakpoint()
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        # breakpoint()
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        # breakpoint()
        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                posterior_mask = torch.cat(
                    [
                        self.vae.encode(
                            imgs[~mask_vae][i: i + 1].to(self.weights_dtype)
                        ).latent_dist.sample()
                        for i in range(imgs.shape[0] - vnum)
                    ],
                    dim=0,
                )
            posterior = torch.cat(
                [
                    self.vae.encode(
                        imgs[mask_vae][i: i + 1].to(self.weights_dtype)
                    ).latent_dist.sample()
                    for i in range(vnum)
                ],
                dim=0,
            )
            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            posterior = self.vae.encode(imgs.permute(1,0,2,3).unsqueeze(0).to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor
            latents = self.patchifier.patchify(latents=latents)

        return latents.to(input_dtype)

    def vae_encode(
            self,
            media_items: Tensor,
            vae: AutoencoderKL,
            split_size: int = 1,
            vae_per_channel_normalize=False,
    ) -> Tensor:
        latents = vae.encode(media_items).latent_dist.sample()

        latents = normalize_latents(latents, vae, vae_per_channel_normalize)
        if is_video_shaped and not isinstance(
                vae, (VideoAutoencoder, CausalVideoAutoencoder)
        ):
            latents = rearrange(latents, "(b n) c h w -> b c n h w", b=batch_size)
        return latents

    @staticmethod
    def normalize_latents(
            latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
    ) -> Tensor:
        return (
            (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
            / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            if vae_per_channel_normalize
            else latents * vae.config.scaling_factor
        )

    @staticmethod
    def un_normalize_latents(
            latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
    ) -> Tensor:
        return (
            latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            if vae_per_channel_normalize
            else latents / vae.config.scaling_factor
        )


    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    def compute_grad_sds(
            self,
            latents: Float[Tensor, "B 4 64 64"],
            text_embeddings: List,
            init_latents: Float[Tensor, "B 4 64 64"],
            latents_mask: Float[Tensor, "B 1 64 64"],
            t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = init_latents * latents_mask[..., None] + latents_noisy * (
                1 - latents_mask[..., None]
            )
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t / 1000] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            alpha = self.scheduler.sigmas[int(t[0])]
            w = (1 - alpha).view(-1, 1, 1, 1).to(noise.device)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        return grad

    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            prompt_utils: PromptProcessorOutput,
            conditioned_image: Float[Tensor, "B H W C"],
            object_noise: Float[Tensor, "B H W"],
            index: int,
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
            camera_distances: Float[Tensor, "B"],
            rgb_as_latents: bool = False,
            num_frames: int = 16,
            **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        batch_size = rgb_BCHW.shape[0] // num_frames
        repeat_factor = num_frames // 8 + 1
        latents: Float[Tensor, "B 4 40 72"]
        if kwargs['train_dynamic_camera']:
            elevation = elevation[[0]]
            azimuth = azimuth[[0]]
            camera_distances = camera_distances[[0]]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (40, 72), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            rgb_BCHW_512 = rgb_BCHW_512.permute(1, 0, 2, 3)[None]
            rgb_BCHW_512 = 2 * rgb_BCHW_512 - 1
            latents = vae_encode(
                rgb_BCHW_512.to(dtype=self.vae.dtype, device=self.vae.device),
                self.vae,
                vae_per_channel_normalize=True,
            ).float()
            latents = self.patchifier.patchify(latents=latents)
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        prompt_attention_mask = text_embeddings[:, 128:, 0]
        text_embeddings = text_embeddings[:, :128]

        conditioned_image = conditioned_image.permute(2, 0, 1).float()
        conditioned_image = conditioned_image.unsqueeze(0).unsqueeze(2)
        conditioned_image = 2 * conditioned_image - 1
        # Encode the input video and repeat to the required number of frame-tokens
        init_latents = vae_encode(
            conditioned_image.to(dtype=self.vae.dtype, device=self.vae.device),
            self.vae,
            vae_per_channel_normalize=True,
        ).float()

        init_latents = init_latents.repeat(1, 1, repeat_factor, 1, 1)
        b, n, f, h, w = init_latents.shape
        conditioning_mask = torch.zeros([b, 1, f, h, w], device=init_latents.device)
        conditioning_mask[:, :, 0] = 1.0
        conditioning_mask = self.patchifier.patchify(conditioning_mask).squeeze(-1)
        init_latents = self.patchifier.patchify(latents=init_latents)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step,
            [batch_size],
            dtype=torch.long,
        )
        t = self.scheduler.timesteps[t].to(self.device)
        grad = self.compute_grad_sds(latents, [text_embeddings, prompt_attention_mask], init_latents,
                                     conditioning_mask, t)
        grad = torch.nan_to_num(grad)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad

        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds_video": loss_sds,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        # t annealing from ProlificDreamer
        if (
                self.cfg.anneal_start_step is not None
                and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )