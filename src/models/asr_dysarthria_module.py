import pytorch_lightning as pl
from transformers import Wav2Vec2ForCTC
from torch.optim import AdamW

class ASRDysarthriaModule(pl.LightningModule):
    def __init__(self, learning_rate=3e-4, vocab_size=None, freeze_feature_extractor=True):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53",
            vocab_size=vocab_size,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        if freeze_feature_extractor:
            self.model.freeze_feature_extractor()

        self.learning_rate = learning_rate

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(input_values=inputs, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(input_values=inputs, labels=labels)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    _ = ASRDysarthriaModule()
