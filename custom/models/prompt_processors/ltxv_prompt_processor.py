import os
from dataclasses import dataclass

import threestudio
import torch
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *
from transformers import T5EncoderModel, T5Tokenizer


@threestudio.register("animate124-ltxv-prompt-processor")
class LTXVPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
    )

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        )

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=128,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            masks = tokens.attention_mask.unsqueeze(2).to(text_encoder.device)
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device),
                                           attention_mask=masks)[0]

            text_embeddings = torch.concatenate((text_embeddings, masks.repeat(1,1,text_embeddings.shape[2])), axis=1)

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
