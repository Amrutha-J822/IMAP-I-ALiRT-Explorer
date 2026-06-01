"""In-memory asynchronous pub/sub broker.

Topics are arbitrary strings; for this project they are instrument names such
as ``"mag"`` or ``"swapi"``. Each subscriber receives its own bounded queue
of ``Message`` objects, so a slow consumer cannot back up the broker - old
messages are dropped on overflow.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    """A single value published to a topic."""

    topic: str
    payload: Any
    sequence: int


@dataclass(eq=False)
class Subscription:
    """One subscriber's view of one or more topics.

    ``eq=False`` keeps the default identity-based hashing so subscriptions
    can be stored in the broker's per-topic sets even though the dataclass
    has mutable fields.
    """

    topics: tuple[str, ...]
    queue: asyncio.Queue[Message] = field(default_factory=lambda: asyncio.Queue(maxsize=256))

    async def receive(self) -> Message:
        return await self.queue.get()


class Broker:
    """Tiny async pub/sub broker.

    Designed for a single-process FastAPI service - good enough to fan-out
    live I-ALiRT samples to dozens of connected WebSocket clients. For
    multi-process deployments this can be swapped for Redis pub/sub or NATS
    without changing the public API.
    """

    def __init__(self, *, max_queue: int = 256) -> None:
        self._max_queue = max_queue
        self._subscriptions: dict[str, set[Subscription]] = {}
        self._latest: dict[str, Message] = {}
        self._sequence: int = 0
        self._lock = asyncio.Lock()

    @property
    def topics(self) -> list[str]:
        return sorted(self._subscriptions)

    def latest(self, topic: str) -> Message | None:
        return self._latest.get(topic)

    async def publish(self, topic: str, payload: Any) -> Message:
        async with self._lock:
            self._sequence += 1
            message = Message(topic=topic, payload=payload, sequence=self._sequence)
            self._latest[topic] = message
            subscribers = list(self._subscriptions.get(topic, ()))

        delivered = 0
        for subscription in subscribers:
            try:
                subscription.queue.put_nowait(message)
                delivered += 1
            except asyncio.QueueFull:
                try:
                    subscription.queue.get_nowait()
                    subscription.queue.put_nowait(message)
                    delivered += 1
                except asyncio.QueueEmpty:  # pragma: no cover - benign race
                    pass

        log.debug(
            "publish topic=%s seq=%d subscribers=%d delivered=%d",
            topic,
            message.sequence,
            len(subscribers),
            delivered,
        )
        return message

    async def _register(self, subscription: Subscription) -> None:
        async with self._lock:
            for topic in subscription.topics:
                self._subscriptions.setdefault(topic, set()).add(subscription)

    async def _unregister(self, subscription: Subscription) -> None:
        async with self._lock:
            for topic in subscription.topics:
                subscribers = self._subscriptions.get(topic)
                if subscribers is None:
                    continue
                subscribers.discard(subscription)
                if not subscribers:
                    self._subscriptions.pop(topic, None)

    @asynccontextmanager
    async def subscribe(self, topics: tuple[str, ...]) -> AsyncIterator[Subscription]:
        subscription = Subscription(
            topics=tuple(topics),
            queue=asyncio.Queue(maxsize=self._max_queue),
        )
        await self._register(subscription)
        try:
            yield subscription
        finally:
            await self._unregister(subscription)
