import threading
from typing import TypeVar

T = TypeVar('T')


class WithMutex:
    def __init__(self, obj: T):
        self.obj = obj
        self.mutex = threading.Lock()
