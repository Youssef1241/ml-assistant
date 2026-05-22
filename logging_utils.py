import logging
import os
from typing import Any


_LOGGING_INITIALIZED = False


def setup_logging(app_name: str = "ml_assistant") -> None:
    """Initialize app logging once per process."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    level_name = os.getenv("ML_ASSISTANT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv(
        "ML_ASSISTANT_LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(stream_handler)

    log_file = os.getenv("ML_ASSISTANT_LOG_FILE", "").strip()
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    logging.getLogger(app_name).info(
        "Logging initialized",
        extra={},
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger, ensuring setup happened first."""
    setup_logging()
    return logging.getLogger(name)


def _serialize(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys[:8]})"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def _repr_truncated(value: Any, max_len: int = 8000) -> str:
    """String form of value for logs; bounded length; safe on repr errors."""
    try:
        s = repr(value)
    except Exception as exc:  # noqa: BLE001
        return f"<repr failed: {exc}>"
    if len(s) > max_len:
        return f"{s[:max_len]}...<truncated total_len={len(s)}>"
    return s


def log_values_with_types(
    logger: logging.Logger,
    level: int,
    message: str,
    **named_values: Any,
) -> None:
    """Log each named value with its fully-qualified type and repr (truncated)."""
    parts: list[str] = []
    for key in sorted(named_values.keys()):
        value = named_values[key]
        t = type(value)
        type_label = f"{t.__module__}.{t.__qualname__}"
        parts.append(f"{key}__type={type_label}")
        parts.append(f"{key}={_repr_truncated(value)}")
    logger.log(level, "%s | %s", message, " ".join(parts))


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log concise context as key=value pairs."""
    if context:
        context_str = " ".join(
            f"{key}={_serialize(value)}" for key, value in sorted(context.items())
        )
        logger.log(level, "%s | %s", message, context_str)
        return
    logger.log(level, message)
