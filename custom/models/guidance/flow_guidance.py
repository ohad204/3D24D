from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffusers import (
#     DDPMScheduler,
#     DPMSolverMultistepScheduler,
#     StableDiffusionPipeline,
#     UNet2DConditionModel,
# )
# from diffusers.loaders import AttnProcsLayers
from diffusers import LTXImageToVideoPipeline
from diffusers import AutoencoderKLLTXVideo
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
# from threestudio.models.prompt_processors.base_sd3 import PromptProcessorOutput_sd3 as PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from custom.utils.image_processor import VaeImageProcessor
from custom.utils.symmetric_patchifier import SymmetricPatchifier
# from diffusers import StableDiffusion3Pipeline
import copy


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("RFDS-sd3")
class RectifiedFlowGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True


        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

        timestep_strategy: str = "uniform"
        num_frames: int = 16
        height: int = 256
        width: int = 256

        video_scale_factor: int = 8

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Rectified Flow ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        @dataclass
        class SubModules:
            pipe: LTXImageToVideoPipeline

        pipe = LTXImageToVideoPipeline.from_pretrained(self.cfg.pretrained_model_name_or_path,
                                                       torch_dtype=torch.float16,
                                                       cache_dir='/data/ohadr')

        pipe = pipe.to(self.device)


        self.submodules = SubModules(pipe=pipe)
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



        del self.pipe.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)


        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        )


        self.scheduler = self.pipe.scheduler
        self.scheduler_sample = self.pipe.scheduler

        self.image_processor = VaeImageProcessor()
        self.patchifier = SymmetricPatchifier(patch_size=1)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        self.noise_scheduler_copy = copy.deepcopy(self.pipe.scheduler)
        threestudio.info(f"Loaded Rectified Flow!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe


    @property
    def transformer(self):
        return self.submodules.pipe.transformer


    @property
    def vae(self):
        return self.submodules.pipe.vae


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe,
        sample_scheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        cross_attention_kwargs = 0,
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_transformer(
        self,
        transformer,
        latents,
        t,
        prompt_embeds,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return transformer(
            hidden_states=latents.to(self.weights_dtype),
            encoder_hidden_states=prompt_embeds[0].to(self.weights_dtype),
            timestep=t.to(self.weights_dtype),
            encoder_attention_mask=prompt_embeds[1].to(self.weights_dtype),
            num_frames=3,
            height=8,
            width=8,
            return_dict=False,
        )[0].to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = torch.clamp(imgs, min=0, max=1)
        imgs = self.image_processor.preprocess(imgs)
        imgs = imgs.permute(1,0,2,3).unsqueeze(0)
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        # print(self.vae.config.shift_factor) 0.06
        # print(self.vae.config.scaling_factor) 1.53
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float16):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_grad_rfds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        init_latents: Float[Tensor, "B 4 64 64"],
        latents_mask: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(latents)
            indices = torch.randint(self.min_step, self.max_step + 1, (B,))
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.device)

            # Add noise according to flow matching.
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            latents_noisy = sigmas * noise + (1.0 - sigmas) * latents
            latents_noisy = torch.concatenate((init_latents[:, :, 0].unsqueeze(2), latents_noisy), dim=2)
            latents_noisy = self.patchifier.patchify(latents=latents_noisy)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            velocity = self.forward_transformer(
                self.transformer,
                latent_model_input,
                torch.cat([timesteps] * 2),
                text_embeddings_vd,
                )
        (
            velocity_text,
            velocity_null,
        ) = velocity.chunk(2)
        u = torch.normal(mean=0, std=1, size=(B,), device=self.device)
        weighting = torch.nn.functional.sigmoid(u)
        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        model_pred = velocity_null + self.cfg.guidance_scale * (velocity_text - velocity_null)
        model_pred = self.patchifier.unpatchify(model_pred, 8, 8, 3, 128)
        return model_pred[:, :, 1:], noise, weighting

    def prepare_conditioning(
        self,
        media_items: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        vae_per_channel_normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the conditioning data for the video generation. If an input media item is provided, encode it
        and set the conditioning_mask to indicate which tokens to condition on. Input media item should have
        the same height and width as the generated video.

        Args:
            media_items (torch.Tensor): media items to condition on (images or videos)
            num_frames (int): number of frames to generate
            height (int): height of the generated video
            width (int): width of the generated video
            method (ConditioningMethod, optional): conditioning method to use. Defaults to ConditioningMethod.UNCONDITIONAL.
            vae_per_channel_normalize (bool, optional): whether to normalize the input to the VAE per channel. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the conditioning latents and the conditioning mask
        """

        # assert media_items.ndim == 5
        assert height == media_items.shape[-2] and width == media_items.shape[-1]

        # Encode the input video and repeat to the required number of frame-tokens
        init_latents = self.encode_images(
            media_items.to(dtype=self.vae.dtype, device=self.vae.device),
            # self.vae,
            # vae_per_channel_normalize=vae_per_channel_normalize,
        ).float()

        init_len, target_len = (
            init_latents.shape[2],
            num_frames // self.cfg.video_scale_factor,
        )
        # if isinstance(self.vae, AutoencoderKLLTXVideo):
        #     target_len += 1
        init_latents = init_latents[:, :, :target_len]
        if target_len > init_len:
            repeat_factor = (target_len + init_len - 1) // init_len  # Ceiling division
            init_latents = init_latents.repeat(1, 1, repeat_factor, 1, 1)[
                :, :, :target_len
            ]

        # Prepare the conditioning mask (1.0 = condition on this token)
        b, n, f, h, w = init_latents.shape
        conditioning_mask = torch.zeros([b, 1, f, h, w], device=init_latents.device)
        conditioning_mask[:, :, 0] = 1.0

        # Patchify the init latents and the mask
        # conditioning_mask = self.patchifier.patchify(conditioning_mask).squeeze(-1)
        # init_latents = self.patchifier.patchify(latents=init_latents)
        return init_latents, conditioning_mask


    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(rgb_BCHW)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils,
        conditioned_image: Float[Tensor, "B H W C"],
        object_noise: Float[Tensor, "B H W"],
        index: int,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)
        # latents = self.patchifier.patchify(latents=latents)

        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=True,
        )
        prompt_attention_mask = text_embeddings_vd[:, 128:, 0]
        text_embeddings = text_embeddings_vd[:, :128]

        media_item = conditioned_image.permute(2,0,1).unsqueeze(0)
        init_latents, conditioning_mask = self.prepare_conditioning(media_item, self.cfg.num_frames,
                                                                    self.cfg.height, self.cfg.width)

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad, noise, weights = self.compute_grad_rfds(
            latents, init_latents, conditioning_mask,
            [text_embeddings, prompt_attention_mask],  camera_condition
        )

        grad = torch.nan_to_num(grad)
        # use reparameterization trick

        target = (grad).detach()
        loss_rfds = weights * F.mse_loss(noise - latents, target, reduction="mean") / batch_size
        return {
            "loss_sds_video": loss_rfds.mean(),
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )