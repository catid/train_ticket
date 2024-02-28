def create_model(type, **kwargs):
    if type == "simple_cnn":
        from .simple_cnn import SimpleCNN
        return SimpleCNN(**kwargs)
    elif type == "simple_mlp":
        from .simple_mlp import SimpleMLP
        return SimpleMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {type}")
