from transformers import ViTConfig, BertConfig
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import BertTokenizerFast

import lightning.pytorch as pl
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

import wandb

class Digitalizer(pl.LightningModule):
    def __init__(self, ):
        super(Digitalizer, self).__init__()

        config_encoder = ViTConfig()
        config_encoder.num_channels = 1
        config_encoder.image_size = (224, 224) # (128, 512)
        config_decoder = BertConfig()
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        
        ## MODELLO
        self.tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-cased")
        self.model = VisionEncoderDecoderModel(config=config)
            # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
            # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
            # set parameters to eventually perform the beam search
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.max_length = 25
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
    
    def configure_optimizers(self):
        learning_rate = 0.0001
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, image, ground_label):
        att_mask = ground_label.attention_mask
        label = ground_label.input_ids  
        out = self.model(pixel_values=image, decoder_attention_mask=att_mask, labels=label)
        return out
    
    def on_train_epoch_start(self):
        self.t_cer = CharErrorRate()
        self.t_wer = WordErrorRate()
    def on_validation_epoch_start(self):
        self.v_cer = CharErrorRate()
        self.v_wer = WordErrorRate()
        
    def training_step(self, batch, batch_idx):
        image = batch["image"].pixel_values                     # ; print("image shape: ", image.shape)
        ground_label = batch["label"].input_ids                 # ; print("labels shape: ", ground_label.shape); pprint(ground_label)
    
        out = self(image=image, ground_label=batch["label"])    # ; print(out.keys())
        logits = out.logits                                     # ; print("logits shape: ", logits.shape)
        loss = out.loss                                         # ; print("loss: ", loss)
        
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            
            pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            ground = ground_label.clone()
            ground[ground == -100] = self.tokenizer.pad_token_id
            ground = self.tokenizer.batch_decode(ground, skip_special_tokens=True)

            cer = self.t_cer(pred, ground)
            wer = self.t_wer(pred, ground)
            
            if batch_idx % 3:
                print("Prediction: "); print(pred)
                print("Ground Truth: "); print(ground)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cer", cer, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wer", wer, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": loss, "train_cer": cer, "train_wer": wer})

        return {"loss": loss, "train_cer": cer, "train_wer": wer}
    
    def validation_step(self, batch, batch_idx):
        image = batch["image"].pixel_values                     # ; print("image shape: ", image.shape)
        ground_label = batch["label"].input_ids                 # ; print("labels shape: ", ground_label.shape); pprint(ground_label)
        
        out = self(image=image, ground_label=batch["label"])    # ; print(out.keys())
        logits = out.logits                                     # ; print("logits shape: ", logits.shape)
        loss = out.loss                                         # ; print("loss: ", loss)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            
            pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True) 
            ground = ground_label.clone()
            ground[ground == -100] = self.tokenizer.pad_token_id
            ground = self.tokenizer.batch_decode(ground, skip_special_tokens=True)
            cer = self.v_cer(pred, ground)
            wer = self.v_wer(pred, ground)
            if batch_idx % 3:
                print("Prediction: "); print(pred)
                print("Ground Truth: "); print(ground)

        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid_cer", cer, on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid_wer", wer, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": loss, "valid_cer": cer, "valid_wer": wer}