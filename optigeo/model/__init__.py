import importlib
from typing import *

if TYPE_CHECKING:
    from .optigeo import OptiGeo

DEFAULT_PRETRAINED_MODEL = "mxliu-hku/OptiGeo"
DEFAULT_PRETRAINED_FILENAME = "OptiGeo.pt"


def import_model_class_by_version(version: Optional[str] = None) -> Type['OptiGeo']:
    if version not in (None, '', 'base', 'optigeo'):
        raise ValueError(f'Unsupported model version: {version}. OptiGeo now uses the base model only.')

    module = importlib.import_module('.optigeo', __package__)

    if not hasattr(module, 'OptiGeo'):
        raise ValueError(f'Class \"OptiGeo\" not found in module {module.__name__}.')

    return getattr(module, 'OptiGeo')
