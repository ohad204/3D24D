import os
from dataclasses import dataclass, field
from easydict import EasyDict as edict
import gc
import tinycudann as tcnn

import numpy as np
import threestudio
import torch
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from .base import BaseLift3DSystem
import torch.nn.functional as F

from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from custom.eval.motion_smoothness import MotionSmoothness
from custom.eval.dynamic_degree import DynamicDegree
from custom.eval.temporal_style import temporal_style


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()

@threestudio.register("animate124-system")
class Animate124(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"

        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)

        guidance_type: str = ""
        guidance: dict = field(default_factory=dict)

        guidance_type_multi_view: str = ""
        guidance_multi_view: dict = field(default_factory=dict)

        prompt_processor_type: str = ""
        prompt_processor: dict = field(default_factory=dict)

        prompt_processor_type_multi_view: str = ""
        prompt_processor_multi_view: dict = field(default_factory=dict)

        ## guidance for ControlNet semantic refine
        guidance_cn_type: str = ""
        guidance_cn: dict = field(default_factory=dict)

        prompt_processor_cn_type: str = ""
        prompt_processor_cn: dict = field(default_factory=dict)
        sr: bool = False # semantic refine

        simultan: bool = False

        time_noise:bool = False

        visualize_samples: bool = False
        prob_multi_view: Optional[float] = None
        prob_single_view_video: Optional[float] = None
        eval_depth_range_perc: Tuple[float, float] = (
            10,
            99,
        )  # Adjust manually based on object, near far depth bounds percentage in (0, 100)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.simultan = self.cfg.get("simultan", False)
        self.static = self.cfg.geometry.pos_encoding_config.get("static", True)
        self.sr = self.cfg.get("sr", False)

        self.guidance = None
        self.prompt_processor = None
        self.prompt_utils = None
        self.geometry_encoding = self.geometry.encoding.encoding

        ## prompt_processor_type_video is T2I if in static stage

        if not self.static:
            self.guidance_video = threestudio.find(self.cfg.guidance_type)(
                self.cfg.guidance
            )
            self.guidance_video.ImageVideo.model.diffusion_model.input_blocks[1][1].transformer_blocks[
                0].attn2.obj_t_clip_idx = self.cfg.prompt_processor.obj_token_clip_idx
            del self.cfg.prompt_processor.obj_token_clip_idx
            self.prompt_processor_video = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.prompt_processor)
            self.prompt_utils_video = self.prompt_processor_video()
            if self.cfg.prob_multi_view not in [0.0, None]:
                self.cfg.prompt_processor_multi_view["prompt"] = self.cfg.prompt_processor["prompt"]
                self.guidance_multi_view = threestudio.find(self.cfg.guidance_type_multi_view)(
                    self.cfg.guidance_multi_view)
                self.prompt_processor_multi_view = threestudio.find(self.cfg.prompt_processor_type_multi_view)(
                    self.cfg.prompt_processor_multi_view
                )
                self.prompt_utils_multi_view = self.prompt_processor_multi_view()

            if self.guidance is None:
                self.guidance = self.guidance_video
                self.prompt_processor = self.prompt_processor_video
                self.prompt_utils = self.prompt_utils_video

        if self.sr:
            self.guidance_cn = threestudio.find(self.cfg.guidance_cn_type)(
                self.cfg.guidance_cn
            )
            prompt_processor_cn_class = threestudio.find(self.cfg.prompt_processor_cn_type)
            self.prompt_processor_cn = prompt_processor_cn_class(self.cfg.prompt_processor_cn)
            self.prompt_utils_cn = self.prompt_processor_cn()
        
        # Set lists to store scores during validation step
        self.val_lpips_iv_scores = [0, 0, 0, 0]
        self.val_lpips_f_scores = [0, 0, 0, 0]
        self.val_motion_smoothness_scores = [0, 0, 0, 0]
        self.val_motion_dynamic_degree = [0, 0, 0, 0]
        self.val_temporal_style_score = [0, 0, 0, 0]
        self.val_clip_tv_scores = [0, 0, 0, 0]
        self.val_clip_iv_scores = [0, 0, 0, 0]
        self.val_clip_f_scores = [0, 0, 0, 0]

        self.start = 0
        self.time_latent = torch.randn(4, 16, 32, 32, device='cuda')
        self.frame_noise_clean = [torch.tensor(i / 16, device='cuda') for i in range(16)]
        self.frame_noise_clean = torch.stack(self.frame_noise_clean)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        if not self.static:
            render_outs = []
            # TODO: Handle batch size higher than 1
            batch["frame_times"] = batch["frame_times"].flatten()
            for frame_idx, frame_time in enumerate(batch["frame_times"].tolist()):
                self.geometry_encoding.frame_time = frame_time
                render_out = self.renderer(**batch)
                if 'validation' not in batch.keys():
                    if frame_idx == 0.0:
                        first_render = render_out
                    render_out = {k: v for k, v in render_out.items() if
                                  k in ["comp_rgb", "depth", "opacity", "static_volume"]}
                else:
                    render_out = {k: v for k, v in render_out.items() if
                                  k in ["comp_rgb", "comp_normal", "opacity", "depth"]}
                render_outs.append(render_out)
            out = {}
            for k in render_out:
                out[k] = torch.cat(
                    [render_out_i[k] for render_out_i in render_outs]
                )
                if 'validation' not in batch.keys():
                    out.update({'first_render': first_render})
        else:
            if 'validation' not in batch.keys():
                render_out = self.renderer(**batch)
                render_out = {k: v for k, v in render_out.items() if k in ["comp_rgb", "depth", "normal", "opacity", "z_variance"]}
                out = {'first_render': render_out}
            else:
                out = self.renderer(**batch)
                out = {k: v for k, v in out.items() if
                              k in ["comp_rgb", "comp_normal", "opacity", "depth"]}

        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def batch_to_one(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and not ('frame_time' in k):
                batch[k] = v[0].unsqueeze(0)
        return batch

    def training_step(self, batch, batch_idx):
        '''
        batch["ref_image"] contains the reference frame information
        '''
        is_video = batch["is_video"]

        if is_video:
            batch = self.batch_to_one(batch)
            static = self.static
            guidance = self.guidance_video
            prompt_utils = self.prompt_utils_video
            self.geometry_encoding.is_video = True
            self.geometry_encoding.set_temp_param_grad(True)
        else:
            static = True
            num_static_frames = 1  # Use a single random time for static guidance
            batch["frame_times"] = batch["frame_times"][
                torch.randperm(batch["frame_times"].shape[0])
            ][:num_static_frames]
            self.geometry_encoding.is_video = False
            self.geometry_encoding.set_temp_param_grad(False)

        out = self(batch)

        if not static:
            guidance_inp = out["comp_rgb"]  # random camera
            if self.cfg.time_noise:
                time_noise = self.frame_noise_clean - batch['frame_times']
                time_noise[1:] += 2 / 16
                alpha = (1 - time_noise).sqrt().view(1, 16, 1, 1)
                beta = time_noise.sqrt().view(1, 16, 1, 1)
            else:
                alpha = (1 - batch['frame_times']).sqrt().view(1, 16, 1, 1)
                beta = batch['frame_times'].sqrt().view(1, 16, 1, 1)
            latent = batch["ref_image"]['object_noise'] * alpha.view(1, 16, 1, 1) + self.time_latent * beta
            guidance_out = guidance(
                guidance_inp, prompt_utils, batch["ref_image"]['rgb'][0], latent,
                batch_idx, **batch, rgb_as_latents=False
            )
            guidance_3d_out = {}
        else:
            guidance_out = {}
            guidance_3d_out = {}
            # guidance_3d_out = self.guidance_multi_view(
            #     guidance_inp, self.prompt_utils_multi_view, **batch, rgb_as_latents=False
            # )

        if self.sr:
            ## controlnet sds
            guidance_cn_out = self.guidance_cn(
                out["comp_rgb"], out["comp_rgb"].detach(), self.prompt_utils_cn, **batch, rgb_as_latents=False
            )
        else:
            guidance_cn_out = {}
            


        loss = 0.0

        for name, value in guidance_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])


        for name, value in guidance_3d_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}_3d", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        for name, value in guidance_cn_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}_cn", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_cn_")]
                )


        if self.cfg.stage == "coarse":
            # Orientation loss proposed in Ref-NeRF. Loss that encourages that all visible normals are facing towards the camera.
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0:
                if "normal" not in out['first_render']:
                    raise ValueError(
                        "comp_normal is required for 2D normal smoothness loss, no comp_normal is found in the output."
                    )
                normal = out['first_render']["normal"]
                loss_normal_smoothness_2d = (
                    normal[:, 1:, :, :] - normal[:, :-1, :, :]
                ).square().mean() + (
                    normal[:, :, 1:, :] - normal[:, :, :-1, :]
                ).square().mean()
                self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
                loss += loss_normal_smoothness_2d * self.C(
                    self.cfg.loss.lambda_normal_smoothness_2d
                )


            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out['first_render']["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out['first_render']["z_variance"][out['first_render']["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            # my anticipate last frame to be similar to the frame before
            if not static:
                times_weight = 1 / torch.diff(batch['frame_times']) / len(batch["frame_times"])
                anticipation_mask = (out["depth"][:-1] > 0)

                if self.C(self.cfg.loss.lambda_anticipate) > 0:
                    anticipation_mask_rgb = anticipation_mask.expand(out["comp_rgb"][:-1].shape)
                    loss_anticipate_rgb = F.l1_loss(out["comp_rgb"][:-1][anticipation_mask_rgb],
                                                    out["comp_rgb"][1:][anticipation_mask_rgb], reduce=False)
                    loss_anticipate_rgb = (loss_anticipate_rgb * times_weight.view(-1, 1, 1, 1)).mean()
                    loss += loss_anticipate_rgb * self.C(self.cfg.loss.lambda_anticipate)

                    loss_anticipate_depth = F.l1_loss(out["depth"][:-1][anticipation_mask],
                                                      out["depth"][1:][anticipation_mask], reduce=False)
                    loss_anticipate_depth = (loss_anticipate_depth * times_weight.view(-1, 1, 1, 1)).mean()
                    loss += loss_anticipate_depth * self.C(self.cfg.loss.lambda_anticipate)

            mask = batch["ref_image"]["mask"]

            if self.static or batch["start_from_zero"]:

                if self.C(self.cfg.loss.lambda_nerf_rgb) > 0:
                    # lower the size of each key in out to size 256x256
                    out['first_render']['comp_rgb'] = F.interpolate(out['first_render']['comp_rgb'].permute(0,3,1,2), (256, 256), mode='bilinear')
                    out['first_render']['comp_rgb'] = out['first_render']['comp_rgb'].permute(0,2,3,1)
                    out['first_render']['depth'] = F.interpolate(out['first_render']['depth'].permute(0,3,1,2), (256, 256), mode='bilinear')
                    out['first_render']['depth'] = out['first_render']['depth'].permute(0,2,3,1)
                    out['first_render']['normal'] = F.interpolate(out['first_render']['normal'].permute(0,3,1,2), (256, 256), mode='bilinear')
                    out['first_render']['normal'] = out['first_render']['normal'].permute(0, 2, 3, 1)
                    out['first_render']['opacity'] = F.interpolate(out['first_render']['opacity'].permute(0,3,1,2), (256, 256), mode='bilinear')
                    out['first_render']['opacity'] = out['first_render']['opacity'].permute(0, 2, 3, 1)

                    loss_nerf_rgb = F.l1_loss(out['first_render']['comp_rgb'][0], batch["ref_image"]['rgb'][0])
                    self.log("train/lambda_nerf_rgb", loss_nerf_rgb)
                    loss += loss_nerf_rgb * self.C(self.cfg.loss.lambda_nerf_rgb)

                    loss_nerf_rgb_fg = F.l1_loss(out['first_render']['comp_rgb'][0:1, :, :, :][mask[..., 0] > 0],
                                               batch["ref_image"]['rgb'][0:1, :, :, :][mask[..., 0] > 0])
                    self.log("train/lambda_nerf_rgb_fg", loss_nerf_rgb_fg)
                    loss += loss_nerf_rgb_fg * self.C(self.cfg.loss.lambda_nerf_rgb)

                if self.C(self.cfg.loss.lambda_nerf_depth) > 0:
                    loss_nerf_depth = F.l1_loss(out['first_render']['depth'][0], batch["ref_image"]['depth'][0])
                    self.log("train/loss_nerf_depth", loss_nerf_depth)
                    loss += loss_nerf_depth * self.C(self.cfg.loss.lambda_nerf_depth)

                if self.C(self.cfg.loss.lambda_nerf_normal) > 0:
                    loss_nerf_normal = F.l1_loss(out['first_render']['normal'][0], batch["ref_image"]['normals'][0])
                    self.log("train/loss_nerf_normal", loss_nerf_normal)
                    loss += loss_nerf_normal * self.C(self.cfg.loss.lambda_nerf_normal)

                if self.C(self.cfg.loss.lambda_nerf_mask) > 0:
                    loss_nerf_mask = F.l1_loss((out['first_render']['depth'][0] > 0).float(),
                                               (batch["ref_image"]['mask'][0] > 0).float())
                    self.log("train/loss_nerf_mask", loss_nerf_mask)
                    loss += loss_nerf_mask * self.C(self.cfg.loss.lambda_nerf_mask)

        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        cleanup()
        batch.update({'validation': True})
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step_img",
            step=self.true_global_step,
        )

        if not self.static:
            with torch.no_grad():

                batch_video = {k: v for k, v in batch.items() if k != "frame_times"}
                # batch_video["frame_times"] = torch.linspace(0.0, 1.0, 16) #batch["frame_times_video"]
                batch_video["frame_times"] = batch["frame_times_video"]
                out_video = self(batch_video)

                # motion smoothness
                motion_smoothness_dfg = 'custom/eval/_AMT/cfgs/AMT-S.yaml'
                motion_smoothness_ckpt = '/home/lab/rahamiob/ohadr/models/AMT/amt-s.pth'
                motion_smoothness = MotionSmoothness(motion_smoothness_dfg, motion_smoothness_ckpt, 'cuda')

                # dynamic degree
                dynamic_degree_ckpt = '/home/lab/rahamiob/ohadr/models/RAFT/models/raft-things.pth'
                args_new = edict(
                    {"model": dynamic_degree_ckpt, "small": False, "mixed_precision": False, "alternate_corr": False})
                dynamic_degree = DynamicDegree(args_new, 'cuda')

                # Calculate LPIPS for current validation iteration
                lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
                ref_image = batch['ref_image']['rgb'].squeeze(0).permute(0, 3, 1, 2)
                FCHW_out_video = out_video['comp_rgb'].permute(0, 3, 1, 2)
                ref_image_batch = ref_image.expand(FCHW_out_video.size(0), -1, -1, -1)
                # before 'aggressive' tensors clamping save variables for CLIP metrics
                # NOTE: use torch.round(FCHW_out_video) for nearest neighbor rounding (for more precision)
                pixels_FCHW_out_video = (FCHW_out_video * 255).clamp(min=0, max=255).to(torch.uint8)
                pixels_ref_image = (ref_image * 255).clamp(min=0, max=255).to(torch.uint8)
                # Clamp tensors to [0,1] - NOTE: we won't lose information since both tensors values is already in [0,1] (still doing clamp just to prevent floating-point bugs)
                FCHW_out_video = FCHW_out_video.clamp(min=0, max=1)
                ref_image_batch = ref_image_batch.clamp(min=0, max=1)
                # LPIPS_IV - calculate the LPIPS between every frame to the reference image
                lpips_iv_score = lpips(FCHW_out_video, ref_image_batch)
                # LPIPS_F - calculate the LPIPS between every two consecutive frames
                lpips.reduction = 'sum'  # change reduction strategy to 'sum' to sum up all lpips
                lpips_f_score = lpips(FCHW_out_video[:-1], FCHW_out_video[1:])
                motion_smoothness_score = motion_smoothness.motion_score(FCHW_out_video)
                dynamic_degree_score = dynamic_degree.infer(FCHW_out_video)
                temporal_style_core = temporal_style(FCHW_out_video, self.cfg.prompt_processor['prompt'])

                # clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").cuda()
                # clip_model, clip_processor = clip_score.model, clip_score.processor
                #
                # processed_vid = clip_processor(text=self.cfg.prompt_processor['prompt'], images=pixels_FCHW_out_video,
                #                                return_tensors="pt", padding=True)  # images: N,C,H,W tensor
                # processed_ref_img = clip_processor(text=self.cfg.prompt_processor['prompt'], images=pixels_ref_image,
                #                                    return_tensors="pt", padding=True)
                #
                # vid_features = clip_model.get_image_features(processed_vid["pixel_values"].to('cuda'))
                # vid_features /= vid_features.norm(p=2, dim=-1, keepdim=True)
                # ref_img_features = clip_model.get_image_features(processed_ref_img["pixel_values"].to('cuda'))
                # ref_img_features /= ref_img_features.norm(p=2, dim=-1, keepdim=True)
                #
                # txt_features = clip_model.get_text_features(
                #     processed_vid["input_ids"].to('cuda'), processed_vid["attention_mask"].to('cuda')
                # )
                # txt_features /= txt_features.norm(p=2, dim=-1, keepdim=True)
                #
                # # CLIP-TV (txt-vid)
                # clip_tv_score = (100 * (vid_features * txt_features).sum(axis=-1)).sum(0)
                # clip_tv_score = torch.max(clip_tv_score / vid_features.shape[0], torch.zeros_like(clip_tv_score))
                # # CLIP-IV (img-vid)
                # clip_iv_score = (100 * (vid_features * ref_img_features).sum(axis=-1)).sum(0)
                # clip_iv_score = torch.max(clip_iv_score / vid_features.shape[0], torch.zeros_like(clip_iv_score))
                # # CLIP-F (vid)
                # clip_f_score = (100 * (vid_features[:-1] * vid_features[1:]).sum(axis=-1)).sum(0)
                # clip_f_score = torch.max(clip_f_score / (vid_features.shape[0] - 1), torch.zeros_like(clip_f_score))

                self.val_lpips_iv_scores[batch_idx] = lpips_iv_score.item()
                self.val_lpips_f_scores[batch_idx] = lpips_f_score.item()
                self.val_motion_smoothness_scores[batch_idx] = motion_smoothness_score
                self.val_motion_dynamic_degree[batch_idx] = dynamic_degree_score
                self.val_temporal_style_score[batch_idx] = temporal_style_core
                # self.val_clip_tv_scores[batch_idx] = clip_tv_score.item()
                # self.val_clip_iv_scores[batch_idx] = clip_iv_score.item()
                # self.val_clip_f_scores[batch_idx] = clip_f_score.item()

                if batch_idx == 3: # TODO: change to last validation index (make it generic)
                    self.log("val/lpips_iv", np.array(self.val_lpips_iv_scores).mean())
                    self.log("val/lpips_f", np.array(self.val_lpips_f_scores).mean())
                    self.log("val/motion_smoothness", np.array(self.val_motion_smoothness_scores).mean())
                    self.log("val/motion_dynamic_degree", np.array(self.val_motion_dynamic_degree).mean())
                    self.log("val/temporal_style", np.array(self.val_temporal_style_score).mean())
                    # self.log("val/clip_tv_score", np.array(self.val_clip_tv_scores).mean())
                    # self.log("val/clip_iv_score", np.array(self.val_clip_iv_scores).mean())
                    # self.log("val/clip_f_score", np.array(self.val_clip_f_scores).mean())

                self.save_image_grid(
                    f"it{self.true_global_step}-{batch['index'][0]}_video.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out_video["comp_rgb"],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                        if "comp_rgb" in out_video
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": out_video["comp_normal"],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                        if "comp_normal" in out_video
                        else []
                    )
                    + [
                        {
                            "type": "grayscale",
                            "img": out_video["opacity"],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ],
                    name="validation_step_video",
                    step=self.true_global_step,
                    video=True,
                    # fps=4,
                )

            if self.cfg.visualize_samples:
                self.save_image_grid(
                    f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                    [
                        {
                            "type": "rgb",
                            "img": self.guidance_single_view.sample(
                                self.prompt_utils_single_view,
                                **batch,
                                seed=self.global_step,
                            )[0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.guidance_single_view.sample_lora(
                                self.prompt_utils_single_view, **batch
                            )[0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ],
                    name="validation_step_samples",
                    step=self.true_global_step,
                )

        cleanup()

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        batch.update({'validation': True})
        if batch_idx == 0:
            self.out_depths = []
        out = self(batch)
        depth = out["depth"][0, :, :, 0].detach().cpu().numpy()
        self.out_depths.append(depth)
        if "comp_rgb" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if "comp_normal" in out:
            self.save_image_grid(
                f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        if not self.static:
            batch_static = {k: v for k, v in batch.items() if k != "frame_times"}
            batch_static["frame_times"] = torch.zeros_like(batch["frame_times"])
            out = self(batch_static)
            if "comp_rgb" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_rgb"][0],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )
            if "comp_normal" in out:
                self.save_image_grid(
                    f"it{self.true_global_step}-test-normal_static/{batch_static['index'][0]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                    ),
                    name="test_step",
                    step=self.true_global_step,
                )

    def on_test_epoch_end(self):
        fps = 15
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=fps,
            name=f"test",
            step=self.true_global_step,
        )
        if not self.static:
            self.save_img_sequence(
                f"it{self.true_global_step}-test_static",
                f"it{self.true_global_step}-test_static",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test_static",
                step=self.true_global_step,
            )
        out_depths = np.stack(self.out_depths)
        non_zeros_depth = out_depths[out_depths != 0]
        self.visu_perc_min_depth = np.percentile(
            non_zeros_depth, self.cfg.eval_depth_range_perc[0]
        )
        self.visu_perc_max_depth = np.percentile(
            non_zeros_depth, self.cfg.eval_depth_range_perc[1]
        )
        depth_color_maps = ["jet"]
        for depth_color_map in depth_color_maps:
            for i, depth in enumerate(out_depths):
                self.save_image_grid(
                    f"it{self.true_global_step}-test-depth-{depth_color_map}/{i}.png",
                    [
                        {
                            "type": "grayscale",
                            "img": depth,
                            "kwargs": {
                                "cmap": depth_color_map,
                                "data_range": "nonzero",
                            },
                        },
                    ],
                    name="depth_test_step",
                    step=self.true_global_step,
                )
        extra_renderings = [
            f"depth-{depth_color_map}" for depth_color_map in depth_color_maps
        ]
        for extra_rendering in extra_renderings:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-{extra_rendering}",
                f"it{self.true_global_step}-test-{extra_rendering}",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name=f"test",
                step=self.true_global_step,
            )
