class STTException(Exception):
    """Base exception for STT-NLP project."""
    pass

class AudioConversionError(STTException):
    """Raised when an audio file conversion fails."""
    pass

class FileOperationError(STTException):
    """Raised when a file operation fails."""
    pass

class EmptyFileError(FileOperationError):
    """Raised when a file is found to be empty."""
    pass

class AudioProcessingError(STTException):
    """Raised when an audio processing step fails."""
    pass
