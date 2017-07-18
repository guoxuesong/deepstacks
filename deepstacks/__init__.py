try:
    import lasagne
except ImportError:
    pass
else:
    from . import lasagne

try:
    import neon
except ImportError:
    pass
else:
    from . import neon

from .stacked import curr_layer,curr_stacks,curr_flags,curr_model
