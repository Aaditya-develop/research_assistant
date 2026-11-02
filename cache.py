import re
import threading
from cachetools import TTLCache

#Exact-match only. A "similar question" cache would happily reuse Q3's numbers for a Q4 question.
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_MAX_ENTRIES = 512

#Routes whose answers don't change between identical questions. Search answers are live, so never cached.
CACHEABLE_ROUTES = {"retrieve", "generate", "out_of_scope"}


def normalize(question: str) -> str:
    #"What were Apple's net sales?" and "what were apple's net sales" should hit the same entry
    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip("?.! ")


class ResponseCache:
    def __init__(self, ttl: int = CACHE_TTL_SECONDS, maxsize: int = CACHE_MAX_ENTRIES):
        self.lock = threading.Lock()
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def key(self, question: str, data_version) -> tuple:
        #data_version changes when documents are ingested, so stale answers are never served
        return (normalize(question), data_version)

    def get(self, question: str, data_version):
        with self.lock:
            return self.cache.get(self.key(question, data_version))

    def set(self, question: str, data_version, response: dict):
        with self.lock:
            self.cache[self.key(question, data_version)] = response

    def clear(self):
        with self.lock:
            self.cache.clear()

    def __len__(self):
        with self.lock:
            return len(self.cache)
