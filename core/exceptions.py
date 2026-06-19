class WebAiError(Exception):
    """所有 WebAi 异常的基类"""
    pass


class SessionExpiredError(WebAiError):
    """token 过期"""
    pass


class BackendNotAvailableError(WebAiError):
    """后端不可用"""
    pass


class PowTimeoutError(WebAiError):
    """PoW 超时"""
    pass


class ResponseTimeoutError(WebAiError):
    """AI 响应超时"""
    pass


class BackendNotFoundError(WebAiError):
    """未知后端"""
    pass
