import numpy as np


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y = [], []

        for i in ind:
            X, Y = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))

        return np.array(x), np.array(y)

    def size(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
