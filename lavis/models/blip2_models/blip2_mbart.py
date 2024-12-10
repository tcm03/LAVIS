import logging

import torch
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import MBartForConditionalGeneration, AutoTokenizer, AutoConfig
from torch import nn

@registry.register_model("blip2_mbart")
class Blip2MBART(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "mbart": "configs/models/blip2/blip2_pretrain_mbart.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        mbart="facebook/mbart-large-50",
        prompt="",
        max_txt_len=32,
    ):
        super().__init__()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.mbart_tokenizer = AutoTokenizer.from_pretrained(mbart)
        self.mbart_model = MBartForConditionalGeneration.from_pretrained(mbart)

        for _, param in self.mbart_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.mbart_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.mbart_model.config.d_model
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_mbart = self.mbart_proj(query_output.last_hidden_state)
        atts_mbart = torch.ones(inputs_mbart.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.mbart_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            output_tokens = self.mbart_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_mbart, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.mbart_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.mbart_model.get_encoder().embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_mbart, inputs_embeds], dim=1)

            outputs = self.mbart_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    # Keep the `generate` and `predict_answers` methods unchanged, except replace `bartpho` with `mbart`.
