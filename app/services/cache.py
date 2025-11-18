"""
Universal async-safe caching layer for RAG systems.
✅ Works with redis.asyncio and sync redis clients
✅ Safe fallbacks to in-memory cache
✅ Handles Redis connection drops automatically
"""

import json
import hashlib
import asyncio
import logging
import time
from typing import Any, Optional, Callable, Dict
from functools import wraps
from collections import OrderedDict
import os
import sys

# ---------------------------------------------------------------------
# Path setup so it works both as package and standalone
# ---------------------------------------------------------------------
try:
    from core.config import settings
except Exception:
    try:
        from ..core.config import settings
    except Exception:
        # fallback for direct execution
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        try:
            from Founders-AI-RAG-Model.core.config import settings
        except Exception:
            class DummySettings:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                redis_max_connections = 10
                cache_ttl_seconds = 600
                cache_max_query_length = 200
            settings = DummySettings()

# ---------------------------------------------------------------------
# Redis import (async-first)
# ---------------------------------------------------------------------
try:
    import redis.asyncio as redis
except ImportError:
    import redis  # fallback to sync

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#  In-memory cache with TTL + LRU
# ---------------------------------------------------------------------
class MemoryCache:
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.ttl_map = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self.cache:
                return None
            if key in self.ttl_map and time.time() > self.ttl_map[key]:
                await self.delete(key)
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
            if ttl:
                self.ttl_map[key] = time.time() + ttl
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            self.cache.pop(key, None)
            self.ttl_map.pop(key, None)
            return True

    async def clear(self) -> None:
        async with self._lock:
            self.cache.clear()
            self.ttl_map.clear()


# ---------------------------------------------------------------------
#  SmartCache (Redis backend + Memory fallback)
# ---------------------------------------------------------------------
class SmartCache:
    def __init__(self):
        self.redis_client: Optional[Any] = None
        self.memory_cache = MemoryCache(max_size=1000)
        self._redis_available = False
        self._lock = asyncio.Lock()
        self._last_check = 0
        self._check_interval = 30  # seconds

    # ------------------------ Helper ------------------------
    async def _safe_await(self, func: Callable):
        try:
            result = func()
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            logger.debug(f"_safe_await failed: {e}")
            return None

    async def initialize(self):
        """Attempt to connect to Redis if URL is provided."""
        if not getattr(settings, "redis_url", None):
            logger.info("Redis not configured; using memory cache only.")
            self._redis_available = False
            return

        async with self._lock:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    max_connections=getattr(settings, "redis_max_connections", 10),
                )
                # Handle both sync & async clients
                ping_result = await self._safe_await(self.redis_client.ping)
                if ping_result is False:
                    raise Exception("Ping failed")
                self._redis_available = True
                logger.info("✅ Redis connected successfully.")
            except Exception as e:
                self.redis_client = None
                self._redis_available = False
                logger.warning(f"Redis unavailable ({e}); using memory cache.")

    async def _should_use_redis(self) -> bool:
        if not getattr(settings, "redis_url", None):
            return False
        if not self._redis_available and (time.time() - self._last_check > self._check_interval):
            self._last_check = time.time()
            await self.initialize()
        return self._redis_available and self.redis_client is not None

    def _make_key(self, prefix: str, identifier: str) -> str:
        if not identifier:
            identifier = "empty"
        if len(identifier) > getattr(settings, "cache_max_query_length", 200):
            identifier = hashlib.sha256(identifier.encode()).hexdigest()
        return f"{prefix}:{identifier}"

    # ------------------------ Core Methods ------------------------
    async def get(self, key: str) -> Optional[Any]:
        if await self._should_use_redis():
            try:
                client = self.redis_client  # <-- FIX
                if client is None:
                    raise RuntimeError("Redis client unexpectedly None")

                val = await self._safe_await(lambda: client.get(key))
                if val is None:
                    return await self.memory_cache.get(key)

                try:
                    return json.loads(val)
                except Exception:
                    return val

            except Exception as e:
                logger.debug(f"Redis GET failed ({e})")

        return await self.memory_cache.get(key)


    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        ttl = ttl or getattr(settings, "cache_ttl_seconds", 600)

        try:
            if await self._should_use_redis():
                client = self.redis_client  
                if client is None:
                    raise RuntimeError("Redis client unexpectedly None")

                await self._safe_await(lambda: client.setex(key, ttl, json.dumps(value, default=str)))
                return True

        except Exception as e:
            logger.debug(f"Redis SET failed ({e})")

        await self.memory_cache.set(key, value, ttl)
        return True


    async def delete(self, key: str) -> bool:
        try:
            if await self._should_use_redis():
                client = self.redis_client   # <-- FIX: promote to local non-optional
                if client is None:
                    raise RuntimeError("Redis client unexpectedly None")

                await self._safe_await(lambda: client.delete(key))
        except Exception as e:
            logger.debug(f"Redis DELETE failed ({e})")

        await self.memory_cache.delete(key)
        return True


    async def delete_pattern(self, pattern: str) -> int:
        deleted = 0

        if await self._should_use_redis():
            client = self.redis_client
            if client is not None:  
                try:
                    async for key in client.scan_iter(match=pattern):
                        await self._safe_await(lambda: client.delete(key))
                        deleted += 1
                except Exception:
                    pass

        await self.memory_cache.clear()
        return deleted

    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        await self.set(key, value, ttl)
        return value

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Async-safe increment with Redis fallback."""
        use_redis = await self._should_use_redis()
        client = self.redis_client  # type narrowing

        if use_redis and client is not None:
            try:
                val = await self._safe_await(lambda: client.incrby(key, amount))
                if val is None:
                    val = 0
                if ttl is not None:
                    await self._safe_await(lambda: client.expire(key, ttl))
                return int(val)
            except Exception as e:
                logger.warning(f"Redis increment failed ({e}), using memory fallback")

        # Memory fallback
        current = await self.memory_cache.get(key) or 0
        new_val = int(current) + amount
        await self.memory_cache.set(key, new_val, ttl)
        return new_val

    async def get_stats(self) -> Dict[str, Any]:
        stats = {"backend": "memory", "size": len(self.memory_cache.cache)}

        try:
            if await self._should_use_redis():
                client = self.redis_client  # <-- FIX: local non-optional reference
                if client is None:
                    raise RuntimeError("Redis client unexpectedly None")

                info = await self._safe_await(lambda: client.info())
                if info:
                    stats.update({
                        "backend": "redis",
                        "clients": info.get("connected_clients", 0),
                        "memory": info.get("used_memory_human", "0B"),
                    })
        except Exception as e:
            logger.debug(f"Redis INFO failed: {e}")

        return stats


# ---------------------------------------------------------------------
# Decorator for async/sync caching
# ---------------------------------------------------------------------
cache = SmartCache()


def cache_result(prefix: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            suffix = key_func(*args, **kwargs) if key_func else "|".join(
                [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()]
            )
            key = cache._make_key(prefix, suffix)
            cached = await cache.get(key)
            if cached is not None:
                return cached

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)

            await cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


# ---------------------------------------------------------------------
# RAG-specific cache utilities
# ---------------------------------------------------------------------
class RAGCache:
    def __init__(self, cache_instance: SmartCache):
        self.cache = cache_instance

    async def cache_rag_query(self,query: str,result: Dict[str, Any],algorithm: str = "hybrid",ttl: Optional[int] = None):
        key = f"rag_query:{algorithm}:{hashlib.sha256(query.encode()).hexdigest()}"
        return await self.cache.set(key, result, ttl)

    async def get_rag_query(self, query: str, algorithm: str = "hybrid") -> Optional[Dict[str, Any]]:
        key = f"rag_query:{algorithm}:{hashlib.sha256((query or 'empty').encode()).hexdigest()}"
        return await self.cache.get(key)

    async def cache_embedding(self, text: str, embedding: list, ttl: int = 3600):
        key = self.cache._make_key("embedding", text)
        return await self.cache.set(key, embedding, ttl)

    async def get_embedding(self, text: str) -> Optional[list]:
        key = self.cache._make_key("embedding", text)
        return await self.cache.get(key)

    async def invalidate_rag_cache(self) -> int:
        total = 0
        for pattern in ["rag_query:*", "embedding:*"]:
            total += await self.cache.delete_pattern(pattern)
        return total


rag_cache = RAGCache(cache)


# ---------------------------------------------------------------------
# Manual Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    async def test():
        await cache.initialize()
        print("Stats before:", await cache.get_stats())
        await cache.set("hello", {"msg": "world"}, ttl=5)
        print("GET hello:", await cache.get("hello"))
        print("Increment test:", await cache.increment("count", 2))
        print("Stats after:", await cache.get_stats())

    asyncio.run(test())
