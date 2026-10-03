"""Custom exceptions used by the floodplain delineation pipeline."""


class ConfigurationError(ValueError):
    """Raised when the YAML configuration is invalid."""


class InputError(ValueError):
    """Raised when an input dataset cannot be used by the workflow."""


class ProcessingError(RuntimeError):
    """Raised when a processing operation fails."""
