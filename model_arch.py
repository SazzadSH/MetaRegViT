from transformers import ViTModel, ViTConfig

# Load pre-trained ViT-Tiny (6M parameters)
config = ViTConfig(
    image_size=224, patch_size=16, num_classes=8,  # Base classes
    hidden_size=128, num_hidden_layers=4, num_attention_heads=4
)
model = ViTModel(config)

# Add dynamic attention masking
class MetaRegViT(nn.Module):
    def __init__(self, vit_model, mask_ratio=0.2):
        super().__init__()
        self.vit = vit_model
        self.mask_ratio = mask_ratio  # 20% heads are trainable
        
    def forward(self, x, task_id=None):
        # Freeze 80% of attention heads, apply task-specific masks
        # (Code for masking selected attention heads here)
        return outputs