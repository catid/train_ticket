def create_model(type, **kwargs):
    if type == "simple_cnn":
        from .simple_cnn import SimpleCNN
        return SimpleCNN(**kwargs)
    elif type == "simple_mlp":
        from .simple_mlp import SimpleMLP
        return SimpleMLP(**kwargs)
    elif type == "vit_tiny":
        from .vit_tiny import ViT
        return ViT(image_size=28, patch_size=4, num_classes=10, dim=512, depth=4, heads=6, mlp_dim=256, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., **kwargs)
    else:
        raise ValueError(f"Unknown model type: {type}")
