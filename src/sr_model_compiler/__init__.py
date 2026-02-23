"""Main functions in the srmodel directory"""

from .utils import call_shell_cmd
from .utils import get_platform_path
from .sr_model_compiler import sr_model_compiler
from .sr_model_compiler import sr_check_model
from .sr_model_compiler import sr_get_compile_log
from .sr100_model_optimizer import sr100_model_optimizer

__all__ = [
    "call_shell_cmd",
    "get_platform_path",
    "sr_model_compiler",
    "sr_check_model",
    "sr_default_config",
    "sr100_model_optimizer",
]
