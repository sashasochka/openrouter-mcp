"""
Multi-Stage Collective Intelligence Orchestrator

This module implements the adaptive, multi-run orchestration flow described in the
product vision statements and documentation updates.  It is responsible for
launching large parallel swarms of heavyweight models, measuring their agreement,
adapting the size of subsequent swarms, and finally delegating the synthesis step
to the single most capable model that is available through OpenRouter.

The implementation deliberately favours readability and explicitness over brevity:
every significant decision point is documented so future contributors (or other AI
agents) can reason about the orchestration behaviour without guessing.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import (
    ModelInfo,
    ModelProvider,
    ProcessingResult,
    TaskContext,
    TaskType,
)
from .consensus_engine import AgreementLevel


@dataclass
class RunConfiguration:
    """
    Operator-facing knobs for the orchestrator.

    The defaults mirror the specification shared in the documentation:
    - Up to four total runs.
    - Start wide (8+ models) and progressively narrow the field on agreement.
    - Expand again if the models disagree.
    """

    max_runs: int = 4
    initial_model_count: int = 8
    max_parallel_models: int = 12
    high_agreement_threshold: float = 0.75
    moderate_agreement_threshold: float = 0.55
    minimum_model_count: int = 2
    timeout_seconds: float = 75.0  # generous safety net for heavyweight models


@dataclass
class RunSnapshot:
    """
    Telemetry for a single orchestration run.

    The snapshot captures everything required to debug or replay decisions later.
    """

    run_index: int
    model_ids: List[str]
    results: List[ProcessingResult]
    agreement_level: AgreementLevel
    average_similarity: float
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalSynthesis:
    """
    Metadata returned alongside the last, high-powered synthesis answer.
    """

    model_id: str
    result: ProcessingResult
    prompt_tokens: int
    summary: str


class MultiStageCollectiveOrchestrator:
    """
    Adaptive orchestrator that runs successive waves of models and manages
    the explore ⇄ exploit feedback loop described in the specification.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        *,
        run_config: Optional[RunConfiguration] = None,
        model_priority: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            model_provider: Abstraction for calling OpenRouter models.
            run_config: Optional overrides for orchestration parameters.
            model_priority: Explicit priority ordering for the final synthesis
                model.  Higher priority entries appear earlier in the sequence.
        """
        self.model_provider = model_provider
        self.config = run_config or RunConfiguration()

        # Priority list used when selecting the "most powerful" model for the
        # final synthesis jump.  Providers mentioned in the brief are weighted
        # heavily, but we keep the list extensible and case-insensitive.
        self.model_priority: Tuple[str, ...] = tuple(
            model_priority
            or (
                "openai/gpt-5-pro",
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.5-pro",
                "xai/grok-4",
                "meta/llama-3.1-400b",
            )
        )

    async def orchestrate(
        self,
        task: TaskContext,
        *,
        candidate_models: Optional[Sequence[str]] = None,
    ) -> Tuple[FinalSynthesis, List[RunSnapshot]]:
        """
        Execute the full multi-stage reasoning pipeline.

        Args:
            task: Canonical task context (problem statement + metadata).
            candidate_models: Optional allowlist of model IDs to consider.

        Returns:
            Tuple where the first element is the final synthesis artefact and
            the second element is the list of intermediate run snapshots.
        """
        # Step 1: discover the model universe.
        available_models = await self.model_provider.get_available_models()
        scored_models = self._score_models(available_models, candidate_models)

        if not scored_models:
            raise ValueError("No models available for multi-stage orchestration.")

        # Seed the first wave with the requested initial count (clamped by the
        # available pool and global parallelism guardrails).
        current_selection = [
            model for model, _score in scored_models[: self.config.initial_model_count]
        ]

        snapshots: List[RunSnapshot] = []
        # Tracks which models have already been used, so that disagreement-driven
        # expansions can introduce fresh perspectives instead of recycling.
        used_model_ids: set[str] = set()

        for round_index in range(1, self.config.max_runs + 1):
            if not current_selection:
                break  # Safety: nothing left to run.

            capped_selection = current_selection[: self.config.max_parallel_models]
            used_model_ids.update(model.model_id for model in capped_selection)

            # Execute the wide parallel pass for this round.
            results = await self._execute_wave(task, capped_selection, round_index)
            agreement, agreement_level = self._measure_agreement(results)

            snapshot = RunSnapshot(
                run_index=round_index,
                model_ids=[result.model_id for result in results],
                results=results,
                agreement_level=agreement_level,
                average_similarity=agreement,
                notes={
                    "selection_size": len(capped_selection),
                    "unique_models_seen": len(used_model_ids),
                },
            )
            snapshots.append(snapshot)

            # Termination criteria:
            # - If we reach high agreement before the last run, we break early.
            # - If we only have a single model left, there is nothing to reduce.
            if (
                agreement_level in (AgreementLevel.UNANIMOUS, AgreementLevel.HIGH_CONSENSUS)
                and round_index >= 2
            ) or len(capped_selection) <= self.config.minimum_model_count:
                break

            # Derive the next wave using the observed agreement.
            current_selection = self._plan_next_wave(
                scored_models,
                used_model_ids,
                results,
                agreement,
                agreement_level,
            )

        # Final step: ask the strongest available model to synthesise the combined answer.
        final_model = self._select_final_model(scored_models, snapshots)
        final_synthesis = await self._generate_final_synthesis(
            task, final_model, snapshots
        )

        return final_synthesis, snapshots

    # ------------------------------------------------------------------
    # Model selection utilities
    # ------------------------------------------------------------------

    def _score_models(
        self,
        models: Iterable[ModelInfo],
        candidate_models: Optional[Sequence[str]],
    ) -> List[Tuple[ModelInfo, float]]:
        """
        Score each model by a coarse "power" metric.

        The heuristic combines provider prestige, context length, and cost-per-token.
        Models not present in the optional allowlist are discarded early.
        """
        allowlist = {model_id.lower() for model_id in candidate_models or []}
        scored: List[Tuple[ModelInfo, float]] = []

        for model in models:
            if allowlist and model.model_id.lower() not in allowlist:
                continue

            provider_score = self._provider_priority(model.provider)
            context_multiplier = math.log(max(model.context_length, 2048), 2) / 10.0
            cost_component = min(model.cost_per_token or 0.00002, 0.0005) * 100.0

            score = provider_score + context_multiplier + cost_component
            scored.append((model, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _provider_priority(self, provider: str) -> float:
        """
        Map provider string to a priority weight.

        Providers highlighted in the specification receive the highest weights.
        """
        provider = (provider or "").lower()
        if "openai" in provider:
            return 5.0
        if "anthropic" in provider:
            return 4.5
        if "google" in provider or "gemini" in provider:
            return 4.2
        if "xai" in provider or "grok" in provider:
            return 4.0
        if "meta" in provider or "llama" in provider:
            return 3.8
        return 3.0  # baseline for other vendors

    # ------------------------------------------------------------------
    # Wave execution helpers
    # ------------------------------------------------------------------

    async def _execute_wave(
        self,
        task: TaskContext,
        models: Sequence[ModelInfo],
        run_index: int,
    ) -> List[ProcessingResult]:
        """
        Fire off a single wave of model calls concurrently.

        Each model receives an augmented TaskContext capturing the current run number
        so downstream logging and telemetry can differentiate outputs.
        """
        async def invoke(model: ModelInfo) -> ProcessingResult:
            # Clone the original task so individual runs remain independent.
            cloned_task = TaskContext(
                task_type=task.task_type,
                content=task.content,
                requirements=dict(task.requirements),
                constraints=dict(task.constraints),
                priority=task.priority,
                deadline=task.deadline,
                metadata={
                    **task.metadata,
                    "orchestration_run": run_index,
                    "orchestration_model": model.model_id,
                },
            )
            return await asyncio.wait_for(
                self.model_provider.process_task(cloned_task, model.model_id),
                timeout=self.config.timeout_seconds,
            )

        tasks = [asyncio.create_task(invoke(model)) for model in models]
        results: List[ProcessingResult] = []

        for task_future in asyncio.as_completed(tasks):
            try:
                result = await task_future
                results.append(result)
            except asyncio.TimeoutError:
                # The orchestrator tolerates slow models by simply omitting them.
                continue
            except Exception as exc:  # pragma: no cover - defensive logging path
                # We do not fail the entire orchestration if a single model call fails.
                continue

        return results

    # ------------------------------------------------------------------
    # Agreement measurement and wave planning
    # ------------------------------------------------------------------

    def _measure_agreement(
        self, results: Sequence[ProcessingResult]
    ) -> Tuple[float, AgreementLevel]:
        """
        Compute a rough agreement metric based on token overlap.

        Heavyweight LLMs often produce richly formatted prose.  For the purposes
        of orchestration we only need a coarse gauge, so we tokenise the outputs
        into lowercase word sets and compute the average pairwise Jaccard score.
        """
        if len(results) < 2:
            return 1.0, AgreementLevel.UNANIMOUS

        # Prepare token sets up-front to avoid repeated work in the nested loop.
        token_sets = [self._normalise_content(result.content) for result in results]

        similarities: List[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a, b = token_sets[i], token_sets[j]
                if not a or not b:
                    continue
                intersection = len(a & b)
                union = len(a | b)
                if union == 0:
                    continue
                similarities.append(intersection / union)

        average_similarity = statistics.mean(similarities) if similarities else 0.0

        if average_similarity >= self.config.high_agreement_threshold:
            level = AgreementLevel.HIGH_CONSENSUS
        elif average_similarity >= self.config.moderate_agreement_threshold:
            level = AgreementLevel.MODERATE_CONSENSUS
        elif average_similarity >= 0.35:
            level = AgreementLevel.LOW_CONSENSUS
        else:
            level = AgreementLevel.NO_CONSENSUS

        return average_similarity, level

    def _normalise_content(self, content: str) -> set[str]:
        """
        Tokenise content into an order-invariant word set.

        Basic normalisation delivers a balance between robustness and speed.
        """
        tokens = []
        current = []
        for char in content.lower():
            if char.isalnum():
                current.append(char)
            elif current:
                tokens.append("".join(current))
                current.clear()
        if current:
            tokens.append("".join(current))
        return set(tokens)

    def _plan_next_wave(
        self,
        scored_models: Sequence[Tuple[ModelInfo, float]],
        used_model_ids: set[str],
        last_results: Sequence[ProcessingResult],
        average_similarity: float,
        agreement_level: AgreementLevel,
    ) -> List[ModelInfo]:
        """
        Decide which models should participate in the next run.

        The policy is intentionally simple:
        - High agreement -> halve the roster (keep the highest-confidence answers).
        - Moderate agreement -> keep roster steady, focusing on confident models.
        - Low agreement -> expand roster with fresh models that have not been used yet.
        """
        current_models = [result.model_id for result in last_results]
        # Sort results by the model-reported confidence so reductions retain
        # the most self-assured answers.
        sorted_results = sorted(
            last_results, key=lambda res: res.confidence, reverse=True
        )

        if agreement_level in (AgreementLevel.HIGH_CONSENSUS, AgreementLevel.UNANIMOUS):
            keep_count = max(self.config.minimum_model_count, len(sorted_results) // 2)
            return self._resolve_models_by_ids(
                [result.model_id for result in sorted_results[:keep_count]],
                scored_models,
            )

        if agreement_level == AgreementLevel.MODERATE_CONSENSUS:
            # Hold the line: keep the same number of models but replace the least
            # confident contributor with a fresh candidate if one exists.
            next_ids = [result.model_id for result in sorted_results]
            replacement_candidates = [
                model for model, _score in scored_models if model.model_id not in used_model_ids
            ]
            if replacement_candidates and next_ids:
                next_ids[-1] = replacement_candidates[0].model_id
                used_model_ids.add(replacement_candidates[0].model_id)
            return self._resolve_models_by_ids(next_ids, scored_models)

        # Low or no consensus: recruit more voices if possible.
        expansion_candidates = [
            model
            for model, _score in scored_models
            if model.model_id not in used_model_ids
        ]
        expanded_ids = current_models.copy()
        for candidate in expansion_candidates:
            if len(expanded_ids) >= self.config.max_parallel_models:
                break
            expanded_ids.append(candidate.model_id)
            used_model_ids.add(candidate.model_id)

        return self._resolve_models_by_ids(expanded_ids, scored_models)

    def _resolve_models_by_ids(
        self,
        model_ids: Sequence[str],
        scored_models: Sequence[Tuple[ModelInfo, float]],
    ) -> List[ModelInfo]:
        """
        Convert a list of model IDs back into ModelInfo references.
        """
        lookup = {model.model_id: model for model, _score in scored_models}
        resolved = [lookup[model_id] for model_id in model_ids if model_id in lookup]
        # Preserve input order but drop duplicates that might appear due to reuse.
        seen = set()
        unique: List[ModelInfo] = []
        for model in resolved:
            if model.model_id in seen:
                continue
            seen.add(model.model_id)
            unique.append(model)
        return unique

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    def _select_final_model(
        self,
        scored_models: Sequence[Tuple[ModelInfo, float]],
        snapshots: Sequence[RunSnapshot],
    ) -> ModelInfo:
        """
        Choose the highest-priority model that actually participated (or is available).

        We give preference to models that have already produced content in earlier
        runs with high confidence, falling back to the static priority ladder.
        """
        # First, search for a participating model that matches the priority ladder.
        participating_lookup = {}
        for snapshot in snapshots:
            for result in snapshot.results:
                participating_lookup[result.model_id] = result

        for preferred_id in self.model_priority:
            if preferred_id in participating_lookup:
                # Exact match found amongst participating models.
                model = next(
                    (model for model, _score in scored_models if model.model_id == preferred_id),
                    None,
                )
                if model:
                    return model

        # Fallback: pick the highest scoring model from the scored list.
        return scored_models[0][0]

    async def _generate_final_synthesis(
        self,
        original_task: TaskContext,
        final_model: ModelInfo,
        snapshots: Sequence[RunSnapshot],
    ) -> FinalSynthesis:
        """
        Compose a meta-prompt summarising all previous runs and ask the
        strongest model to deliver the final answer.
        """
        summary_lines: List[str] = []
        for snapshot in snapshots:
            header = (
                f"Run {snapshot.run_index}: agreement {snapshot.agreement_level.value} "
                f"(avg similarity {snapshot.average_similarity:.2f})"
            )
            summary_lines.append(header)

            for result in snapshot.results:
                truncated = (result.content or "").strip().replace("\n", " ")
                truncated = truncated[:400] + ("…" if len(truncated) > 400 else "")
                summary_lines.append(
                    f"- {result.model_id}: confidence {result.confidence:.2f} → {truncated}"
                )

        synthesis_prompt = (
            "You are the final-stage arbiter in a multi-model deliberation. "
            "Several heavyweight models have already analysed the problem. "
            "Your task is to read their findings, weigh arguments, resolve conflicts, "
            "and produce the single best answer possible. "
            "Always acknowledge major points of agreement, address disagreements, "
            "and integrate the strongest reasoning threads. "
            "\n\n"
            "Problem statement:\n"
            f"{original_task.content.strip()}\n\n"
            "Deliberation transcript:\n"
            + "\n".join(summary_lines)
            + "\n\n"
            "Final response requirements:\n"
            "- Provide a concise executive summary first.\n"
            "- Follow with a detailed synthesis that references supporting models.\n"
            "- Highlight any residual uncertainty and suggested follow-up analysis.\n"
        )

        synthesis_task = TaskContext(
            task_type=TaskType.REASONING,
            content=synthesis_prompt,
            requirements={
                "system_prompt": (
                    "You are assembling the definitive answer by combining the strengths "
                    "of the deliberation above. Cite contributing models inline (e.g. "
                    "\"(Claude)\" or \"(Gemini)\"). Keep the tone confident yet honest "
                    "about uncertainties."
                )
            },
            metadata={
                "orchestration_phase": "final_synthesis",
                "deliberation_runs": len(snapshots),
                "candidate_models": [snapshots[0].model_ids if snapshots else []],
            },
        )

        final_result = await self.model_provider.process_task(
            synthesis_task, final_model.model_id
        )

        return FinalSynthesis(
            model_id=final_model.model_id,
            result=final_result,
            prompt_tokens=len(synthesis_prompt.split()),
            summary="\n".join(summary_lines),
        )

