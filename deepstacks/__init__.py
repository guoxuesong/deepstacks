from .stacked import curr_layer, curr_stacks, curr_flags, curr_model
from .stacked import set_verbose

#__all__ = []

try:
    import lasagne
except ImportError:
    pass
#else:
#    __all__ += [lasagne]

try:
    import neon
except ImportError:
    pass
#else:
#    __all__ += [neon]

#__all__ += [curr_layer, curr_stacks, curr_flags, curr_model]
