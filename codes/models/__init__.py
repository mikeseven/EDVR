import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'VideoSR_base':
        from .VideoSR_base_model import VideoSRBaseModel as M
    else:
        raise NotImplementedError(f'Model [{model}] not recognized.')
    m = M(opt)
    logger.info(f'Model [{m.__class__.__name__}] is created.')
    return m
