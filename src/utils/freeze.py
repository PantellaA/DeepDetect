def set_trainable_head_only(model):
    for name, p in model.named_parameters():
        p.requires_grad = ("fc" in name)

def set_trainable_layer4_and_head(model):
    for name, p in model.named_parameters():
        p.requires_grad = (("layer4" in name) or ("fc" in name))
