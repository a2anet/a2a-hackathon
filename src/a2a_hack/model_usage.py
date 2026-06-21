"""Normalized model usage accounting for hackathon runs."""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field
from tau2.data_model.message import Message, UserMessage


# Controlled marking models.
ANTHROPIC_SONNET_INPUT_USD_PER_MILLION = 3.0
ANTHROPIC_SONNET_OUTPUT_USD_PER_MILLION = 15.0
ANTHROPIC_CACHE_READ_MULTIPLIER = 0.1
ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25


class ModelUsageRecord(BaseModel):
    """One normalized model call usage record."""

    actor: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    estimated_cost_usd: float | None = None


class ModelUsageTotals(BaseModel):
    """Aggregated model usage counters."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    models: dict[str, "ModelUsageTotals"] = Field(default_factory=dict)
    actors: dict[str, "ModelUsageTotals"] = Field(default_factory=dict)


def estimate_model_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> float:
    """Estimate USD cost for a normalized usage record."""
    if not _is_anthropic_sonnet(model):
        return 0.0

    input_rate = ANTHROPIC_SONNET_INPUT_USD_PER_MILLION
    output_rate = ANTHROPIC_SONNET_OUTPUT_USD_PER_MILLION
    uncached_input_tokens = max(
        input_tokens - cache_read_input_tokens - cache_write_input_tokens,
        0,
    )
    return (
        uncached_input_tokens * input_rate
        + output_tokens * output_rate
        + cache_read_input_tokens * input_rate * ANTHROPIC_CACHE_READ_MULTIPLIER
        + cache_write_input_tokens * input_rate * ANTHROPIC_CACHE_WRITE_MULTIPLIER
    ) / 1_000_000


def usage_records_from_messages(
    messages: Iterable[Message],
    actor: str = "user_simulator",
) -> list[ModelUsageRecord]:
    """Extract normalized usage records from tau2 messages."""
    records: list[ModelUsageRecord] = []
    for message in messages:
        if not isinstance(message, UserMessage) or message.usage is None:
            continue
        raw_data = message.raw_data if isinstance(message.raw_data, dict) else {}
        raw_usage_data = raw_data.get("usage")
        raw_usage = raw_usage_data if isinstance(raw_usage_data, dict) else {}
        usage = message.usage
        model = _string(raw_data.get("model")) or "unknown"
        input_tokens = _int(usage.get("prompt_tokens"))
        output_tokens = _int(usage.get("completion_tokens"))
        cache_read_input_tokens = _cache_read_tokens(raw_usage)
        cache_write_input_tokens = _cache_write_tokens(raw_usage)
        cost = message.cost if message.cost and message.cost > 0 else None
        records.append(
            ModelUsageRecord(
                actor=actor,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_write_input_tokens=cache_write_input_tokens,
                estimated_cost_usd=cost,
            )
        )
    return records


def aggregate_model_usage(records: Iterable[ModelUsageRecord]) -> dict[str, Any]:
    """Aggregate usage records into run-document model_usage."""
    totals = ModelUsageTotals()
    for record in records:
        _add_record(totals, record)
        _add_record(
            totals.models.setdefault(record.model, ModelUsageTotals()),
            record,
        )
        _add_record(
            totals.actors.setdefault(record.actor, ModelUsageTotals()),
            record,
        )
    return _totals_to_dict(totals)


def _add_record(totals: ModelUsageTotals, record: ModelUsageRecord) -> None:
    totals.calls += 1
    totals.input_tokens += record.input_tokens
    totals.output_tokens += record.output_tokens
    totals.cache_read_input_tokens += record.cache_read_input_tokens
    totals.cache_write_input_tokens += record.cache_write_input_tokens
    if _is_anthropic_sonnet(record.model):
        totals.estimated_cost_usd += estimate_model_cost_usd(
            record.model,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_input_tokens,
            record.cache_write_input_tokens,
        )
    elif record.estimated_cost_usd is not None:
        totals.estimated_cost_usd += record.estimated_cost_usd


def _totals_to_dict(totals: ModelUsageTotals) -> dict[str, Any]:
    data = totals.model_dump(exclude={"models", "actors"})
    data["estimated_cost_usd"] = round(totals.estimated_cost_usd, 8)
    if totals.models:
        data["models"] = {
            model: _totals_to_dict(model_totals)
            for model, model_totals in sorted(totals.models.items())
        }
    if totals.actors:
        data["actors"] = {
            actor: _totals_to_dict(actor_totals)
            for actor, actor_totals in sorted(totals.actors.items())
        }
    return data


def _cache_read_tokens(raw_usage: dict[str, Any]) -> int:
    details = raw_usage.get("prompt_tokens_details")
    input_details = raw_usage.get("input_tokens_details")
    return max(
        _int(raw_usage.get("cache_read_input_tokens")),
        _int(raw_usage.get("cached_tokens")),
        _int(details.get("cached_tokens") if isinstance(details, dict) else None),
        _int(input_details.get("cached_tokens") if isinstance(input_details, dict) else None),
    )


def _is_anthropic_sonnet(model: str) -> bool:
    normalized_model = model.lower()
    return "claude" in normalized_model and "sonnet" in normalized_model


def _cache_write_tokens(raw_usage: dict[str, Any]) -> int:
    return max(
        _int(raw_usage.get("cache_creation_input_tokens")),
        _int(raw_usage.get("cache_write_input_tokens")),
    )


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""
