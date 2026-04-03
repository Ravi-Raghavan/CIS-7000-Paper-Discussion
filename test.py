import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style) for time-series forecasting.
    """
    def __init__(self, context_len=20, forecast_len=30,
                 d_model=128, nhead=4, num_layers=3, patch_size=32):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = context_len // patch_size
        self.forecast_len = forecast_len
        self.d_model = d_model

        # ---- Patch embedding ----
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_emb     = nn.Embedding(self.num_patches, d_model)

        # ---- Decoder-only stack ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ---- Output head ----
        self.head = nn.Linear(d_model * self.num_patches, forecast_len)

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device),
            diagonal=1
        )

    def forward(self, x):
        """
        x: (B, context_len)
        """
        B = x.size(0)
        device = x.device

        # ---- Patchify ----
        patches = x.view(B, self.num_patches, self.patch_size)  # (B, P, patch_size)
        h = self.patch_embed(patches)                           # (B, P, d_model)

        # ---- Add positional encoding ----
        pos = torch.arange(self.num_patches, device=device)
        h = h + self.pos_emb(pos)

        # ---- Causal mask ----
        tgt_mask = self._causal_mask(self.num_patches, device)

        # ---- Decoder (no encoder memory) ----
        # memory=None → acts like GPT-style self-attention
        h = self.decoder(tgt=h, memory=h, tgt_mask=tgt_mask)

        # ---- Flatten + predict ----
        h = h.reshape(B, -1)
        return self.head(h)
    
model = TimeSeriesTransformer()
for name, module in list(model.decoder.named_modules()):
    if isinstance(module, nn.Linear):
        print(name, module)
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.decoder
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)