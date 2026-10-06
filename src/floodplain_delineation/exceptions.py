"""Exception hierarchy used across the package.

All expected failures derive from :class:`FloodplainError` so the command-line
interface can report them cleanly (message + non-zero exit status) while still
letting genuinely unexpected exceptions surface with a traceback.
"""


class FloodplainError(Exception):
    """Base class for all expected, user-reportable errors."""


class ConfigError(FloodplainError):
    """The YAML configuration is missing, malformed, or contains invalid values."""


class InputError(FloodplainError):
    """An input dataset (DEM or pilot area) is missing, unreadable, or unusable."""


class ProcessingError(FloodplainError):
    """A processing stage failed or produced an unusable result."""


class WhiteboxError(ProcessingError):
    """A WhiteboxTools invocation failed or produced a missing/corrupt output."""


class OutputError(FloodplainError):
    """Output directory, overwrite, or file-writing problem."""
