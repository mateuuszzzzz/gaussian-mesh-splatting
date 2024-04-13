from .backbones import Encoder, Sphere2ModelDecoder, Sphere2ModelTargetNetwork, Face2GSParamsDecoder, Face2GSParamsTargetNetwork

TRAINABLE = {Face2GSParamsDecoder, Face2GSParamsTargetNetwork}
PRETRAINED = {Encoder, Sphere2ModelDecoder, Sphere2ModelTargetNetwork}

def hypercloud_training():
    # TO DO: Implement training loop
    pass