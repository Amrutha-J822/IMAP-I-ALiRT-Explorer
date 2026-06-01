from __future__ import annotations

import asyncio

import pytest

from ialirt_explorer.service.pubsub import Broker


@pytest.mark.asyncio
async def test_broker_publishes_to_matching_subscribers() -> None:
    broker = Broker(max_queue=8)

    async with broker.subscribe(("mag",)) as subscription:
        await broker.publish("mag", {"value": 1.0})
        message = await asyncio.wait_for(subscription.receive(), timeout=1.0)

    assert message.topic == "mag"
    assert message.payload == {"value": 1.0}
    assert message.sequence == 1


@pytest.mark.asyncio
async def test_broker_ignores_other_topics() -> None:
    broker = Broker()

    async with broker.subscribe(("mag",)) as subscription:
        await broker.publish("swapi", {"value": 99})
        await broker.publish("mag", {"value": 1.0})
        message = await asyncio.wait_for(subscription.receive(), timeout=1.0)

    assert message.topic == "mag"
    assert message.payload["value"] == 1.0


@pytest.mark.asyncio
async def test_broker_drops_oldest_message_when_queue_full() -> None:
    broker = Broker(max_queue=2)

    async with broker.subscribe(("mag",)) as subscription:
        for value in range(5):
            await broker.publish("mag", {"value": value})

        received: list[int] = []
        for _ in range(2):
            message = await asyncio.wait_for(subscription.receive(), timeout=1.0)
            received.append(int(message.payload["value"]))

    assert received == [3, 4]


@pytest.mark.asyncio
async def test_broker_latest_returns_most_recent_message() -> None:
    broker = Broker()
    await broker.publish("mag", {"value": 1})
    await broker.publish("mag", {"value": 2})

    latest = broker.latest("mag")
    assert latest is not None
    assert latest.payload["value"] == 2


@pytest.mark.asyncio
async def test_broker_unregisters_subscription_on_exit() -> None:
    broker = Broker()
    async with broker.subscribe(("mag",)):
        assert "mag" in broker.topics
    assert "mag" not in broker.topics
