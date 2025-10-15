import pytest

from src.openrouter_mcp.collective_intelligence.base import ProcessingResult, TaskContext, TaskType
from src.openrouter_mcp.collective_intelligence.consensus_engine import AgreementLevel
from src.openrouter_mcp.collective_intelligence.orchestrator import FinalSynthesis, RunSnapshot
from src.openrouter_mcp.handlers.collective_intelligence import (
    MultiStageCollectiveRequest,
    multi_stage_collective_answer,
)


class DummyClient:
    """Simple async context manager used to patch the OpenRouter client helper."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeOrchestrator:
    """Orchestrator stub that returns canned results for handler testing."""

    def __init__(self, model_provider, run_config):
        self.model_provider = model_provider
        self.run_config = run_config
        self.invocations = 0

    async def orchestrate(self, task: TaskContext, candidate_models=None):
        self.invocations += 1
        # Mimic final synthesis
        final_result = ProcessingResult(
            task_id=task.task_id,
            model_id="openai/gpt-5-pro",
            content="Synthesised answer from multiple heavyweight models.",
            confidence=0.96,
            processing_time=2.4,
            tokens_used=420,
            cost=0.62,
        )
        final = FinalSynthesis(
            model_id="openai/gpt-5-pro",
            result=final_result,
            prompt_tokens=512,
            summary="Run 1: high_consensus -> Assembled solution.",
        )

        snapshot_result = ProcessingResult(
            task_id=task.task_id,
            model_id="google/gemini-2.5-pro",
            content="Gemini perspective on the task.",
            confidence=0.84,
            processing_time=1.1,
            tokens_used=300,
            cost=0.31,
        )

        snapshot = RunSnapshot(
            run_index=1,
            model_ids=[snapshot_result.model_id],
            results=[snapshot_result],
            agreement_level=AgreementLevel.HIGH_CONSENSUS,
            average_similarity=0.82,
        )

        return final, [snapshot]


@pytest.mark.asyncio
async def test_multi_stage_collective_handler(monkeypatch):
    """
    Ensure the MCP tool surfaces orchestrator output with the expected schema.
    """

    # Patch OpenRouter client factory to avoid network usage.
    from src.openrouter_mcp.handlers import collective_intelligence as ci

    monkeypatch.setattr(ci, "get_openrouter_client", lambda: DummyClient())
    monkeypatch.setattr(ci, "OpenRouterModelProvider", lambda client: object())
    monkeypatch.setattr(ci, "MultiStageCollectiveOrchestrator", FakeOrchestrator)

    request = MultiStageCollectiveRequest(
        prompt="Determine the optimal AI model deployment strategy.",
        task_type="analysis",
        initial_model_count=4,
        max_runs=2,
        max_parallel_models=5,
    )

    tool = multi_stage_collective_answer
    response = await tool.fn(request)

    assert response["final_model"] == "openai/gpt-5-pro"
    assert response["final_confidence"] == pytest.approx(0.96, rel=1e-6)
    assert response["run_summaries"], "Run summaries should be populated."
    summary = response["run_summaries"][0]
    assert summary["agreement_level"] == AgreementLevel.HIGH_CONSENSUS.value
    assert response["deliberation_summary"].startswith("Run 1")
