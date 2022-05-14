import threading


class MutexMap:
    def __init__(self, map):
        self.map = map
        self.mutex = threading.Lock()

    def __getitem__(self, key):
        with self.mutex:
            return self.map[key]

    def __setitem__(self, key, value):
        with self.mutex:
            self.map[key] = value

    def __contains__(self, key):
        with self.mutex:
            return key in self.map

    def __len__(self):
        with self.mutex:
            return len(self.map)

    def __iter__(self):
        with self.mutex:
            return iter(self.map)

    def incr(self, key, value=1):
        with self.mutex:
            self.map[key] += value
