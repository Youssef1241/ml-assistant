from processing.encoding_handling import *
combinations_config = {
    "encoding": {
        "method": handle_encoding,
    },
    "skew": {
        "method": handle_skew,
    },
    "scaling": {
        "method": handle_scaling,
    },
    "imbalance_methods": {
        "method": handle_imbalance,
    }
}