import pytest

from src.openrouter_mcp.collective_intelligence.orchestrator import (
    MultiStageCollectiveOrchestrator,
    RunConfiguration,
)
from src.openrouter_mcp.collective_intelligence.base import (
    ModelCapability,
    ModelInfo,
    ProcessingResult,
    TaskContext,
    TaskType,
)
from src.openrouter_mcp.collective_intelligence.consensus_engine import AgreementLevel


class FakeModelProvider:
    """
    Test double that emulates OpenRouter responses without any network traffic.

    Each wave (run) returns canned responses per model so we can precisely model
    agreement / disagreement scenarios.  The final synthesis call is also stubbed.
    """

    def __init__(self, wave_outputs, final_model="openai/gpt-5-pro"):
        self.wave_outputs = wave_outputs
        self.final_model = final_model
        self.final_calls = 0

        self.models = [
            ModelInfo(
                model_id="openai/gpt-5-pro",
                name="GPT-5 Pro",
                provider="OpenAI",
                capabilities={ModelCapability.REASONING: 0.95},
                context_length=32768,
                cost_per_token=0.00008,
            ),
            ModelInfo(
                model_id="anthropic/claude-3.5-sonnet",
                name="Claude 3.5",
                provider="Anthropic",
                capabilities={ModelCapability.REASONING: 0.92},
                context_length=200000,
                cost_per_token=0.00006,
            ),
            ModelInfo(
                model_id="google/gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                provider="Google",
                capabilities={ModelCapability.REASONING: 0.9},
                context_length=1000000,
                cost_per_token=0.00005,
            ),
            ModelInfo(
                model_id="xai/grok-4",
                name="Grok 4",
                provider="xAI",
                capabilities={ModelCapability.REASONING: 0.88},
                context_length=65536,
                cost_per_token=0.000045,
            ),
            ModelInfo(
                model_id="meta/llama-3.1-400b",
                name="Llama 3.1 400B",
                provider="Meta",
                capabilities={ModelCapability.REASONING: 0.86},
                context_length=200000,
                cost_per_token=0.00004,
            ),
        ]

    async def get_available_models(self):
        return self.models

    async def process_task(self, task: TaskContext, model_id: str, **_) -> ProcessingResult:
        # Final synthesis requests advertise the orchestration phase.
        if task.metadata.get("orchestration_phase") == "final_synthesis":
            self.final_calls += 1
            return ProcessingResult(
                task_id=task.task_id,
                model_id=model_id,
                content=f"[FINAL] Consolidated answer for: {task.content[:60]}",
                confidence=0.97,
                processing_time=1.2,
                tokens_used=512,
                cost=0.48,
            )

        run_index = task.metadata.get("orchestration_run", 0)
        content, confidence = self.wave_outputs.get(run_index, {}).get(
            model_id, (f"default response {model_id}", 0.7)
        )
        return ProcessingResult(
            task_id=task.task_id,
            model_id=model_id,
            content=content,
            confidence=confidence,
            processing_time=0.8,
            tokens_used=256,
            cost=0.12,
        )


def _high_agreement_wave():
    shared = "Shared collective insight about renewable microgrids"
    return {
        1: {
            "openai/gpt-5-pro": (shared, 0.92),
            "anthropic/claude-3.5-sonnet": (shared, 0.9),
            "google/gemini-2.5-pro": (shared, 0.91),
        }
    }


def _low_agreement_wave():
    return {
        1: {
            "openai/gpt-5-pro": ("Solar focus plan with aggressive storage", 0.82),
            "anthropic/claude-3.5-sonnet": ("Geothermal-first proposal for reliability", 0.8),
            "google/gemini-2.5-pro": ("Offshore wind centric approach", 0.79),
        },
        2: {
            "openai/gpt-5-pro": ("Refined solar + storage blend", 0.88),
            "anthropic/claude-3.5-sonnet": ("Consensus leaning toward mixed portfolio", 0.86),
            "google/gemini-2.5-pro": ("Supporting wind + solar integration", 0.84),
            "xai/grok-4": ("Adds nuclear perspective for baseload", 0.8),
            "meta/llama-3.1-400b": ("Emphasises demand-response and grid orchestration", 0.78),
        },
    }


@pytest.mark.asyncio
async def test_orchestrator_shrinks_population_after_high_agreement():
    provider = FakeModelProvider(_high_agreement_wave())
    orchestrator = MultiStageCollectiveOrchestrator(
        provider,
        run_config=RunConfiguration(
            max_runs=4,
            initial_model_count=3,
            max_parallel_models=3,
        ),
    )

    task = TaskContext(
        task_type=TaskType.REASONING,
        content="Evaluate city-wide renewable energy strategy.",
    )

    final, snapshots, progress_events = await orchestrator.orchestrate(task)

    assert len(snapshots) >= 2, "High consensus should trigger a follow-up run with fewer agents."
    first, second = snapshots[0], snapshots[1]
    assert first.agreement_level == AgreementLevel.HIGH_CONSENSUS
    assert len(second.model_ids) < len(first.model_ids), "Roster should shrink after agreement."
    assert final.model_id == "openai/gpt-5-pro"
    assert provider.final_calls == 1
    assert "[FINAL]" in final.result.content
    assert progress_events, "Progress events should be recorded for visibility."


@pytest.mark.asyncio
async def test_orchestrator_expands_on_low_agreement():
    provider = FakeModelProvider(_low_agreement_wave())
    orchestrator = MultiStageCollectiveOrchestrator(
        provider,
        run_config=RunConfiguration(
            max_runs=3,
            initial_model_count=3,
            max_parallel_models=5,
        ),
    )

    task = TaskContext(
        task_type=TaskType.REASONING,
        content="Plan resilient regional energy infrastructure.",
    )

    final, snapshots, progress_events = await orchestrator.orchestrate(task)

    assert len(snapshots) >= 2, "Low agreement should trigger additional runs."
    first_models = set(snapshots[0].model_ids)
    second_models = set(snapshots[1].model_ids)
    assert second_models - first_models, "Additional models should be introduced on low agreement."
    assert final.model_id == "openai/gpt-5-pro"
    assert provider.final_calls == 1
    assert progress_events, "Progress events should be recorded for visibility."
