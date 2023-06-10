def creat_model(args, body_parts, datasets, topology_name):
    if args.architecture_name == 'pan':
        import models.architecture_humdog
        return models.architecture_humdog.PAN_model(args, body_parts, datasets, topology_name)
    else:
        raise Exception('Unimplemented model')


def create_model_mixamo(args, character_names, dataset):
    if args.model == 'pan':
        import models.architecture_mixamo
        return models.architecture_mixamo.PAN_model(args, character_names, dataset)
    else:
        raise Exception('Unimplemented model')