def create_model(type, **kwargs):
    if type == "simple_cnn":
        from .simple_cnn import SimpleCNN
        return SimpleCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {type}")
