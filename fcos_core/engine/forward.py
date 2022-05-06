from fcos_core.structures.image_list import to_image_list

def foward_detector(model, images, targets=None, return_maps=False):
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    images = to_image_list(images)
    features = model_backbone(images.tensors)

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    losses = {}

    if model_fcos.training and targets is None:
        # train G on target domain
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=None, return_maps=return_maps)
        assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0 ,"proposal losses:{}".format(proposal_losses["zero"])# loss_dict should be empty dict
    else:
        # train G on source domain / inference
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

    if model_fcos.training:
        # training
        m = {
            layer: {
                map_type:
                score_maps[map_type][map_layer_to_index[layer]]
                for map_type in score_maps
            }
            for layer in feature_layers
        }
        losses.update(proposal_losses)
        return losses, f, m
    else:
        # inference
        result = proposals
        return result


def foward_detector_roifeature(model, images, targets=None, return_maps=False):
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()

    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    images = to_image_list(images)
    features = model_backbone(images.tensors)

    f = {
        layer: features[map_layer_to_index[layer]]
        for layer in feature_layers
    }
    losses = {}

    if model_fcos.training and targets is None:
        # train G on target domain
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=None, return_maps=return_maps)
        assert len(proposal_losses) == 1 and proposal_losses["zero"] == 0 ,"proposal losses:{}".format(proposal_losses["zero"])# loss_dict should be empty dict
    else:
        # train G on source domain / inference
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

    if model_fcos.training:
        # training
        m = {
            layer: {
                map_type:
                score_maps[map_type][map_layer_to_index[layer]]
                for map_type in score_maps
            }
            for layer in feature_layers
        }
        losses.update(proposal_losses)
        return losses, f, m
    else:
        # inference
        result = proposals
        return result