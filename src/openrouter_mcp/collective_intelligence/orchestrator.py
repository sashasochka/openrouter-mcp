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
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .base import (
    ModelInfo,
    ModelProvider,
    ProcessingResult,
    TaskContext,
    TaskType,
)
from .consensus_engine import AgreementLevel

if TYPE_CHECKING:
    from fastmcp.server.context import Context


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
    timeout_seconds: float = 3600.0  # GPT-5-class models can stream for up to an hour


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
class ModelPlan:
    """Desired default composition of the orchestration roster."""

    model_id: str
    count: int = 1
    request_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInvocation:
    """Concrete model invocation including duplicate instances and options."""

    model: ModelInfo
    plan: ModelPlan
    instance_index: int = 0

    @property
    def label(self) -> str:
        return f"{self.model.model_id}#{self.instance_index + 1}"

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
        default_plan: Optional[Sequence[ModelPlan]] = None,
        final_synthesis_model_id: Optional[str] = None,
        final_synthesis_options: Optional[Dict[str, Any]] = None,
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

        self.default_plan: Tuple[ModelPlan, ...] = tuple(default_plan or ())
        self.final_synthesis_model_id = final_synthesis_model_id
        self.final_synthesis_options = final_synthesis_options or {}

    async def orchestrate(
        self,
        task: TaskContext,
        *,
        candidate_models: Optional[Sequence[str]] = None,
        context: Optional["Context"] = None,
    ) -> Tuple[FinalSynthesis, List[RunSnapshot], List[Dict[str, Any]]]:
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
        invocation_pool = self._build_invocations(available_models, candidate_models)
        scored_models = self._score_invocations(invocation_pool)

        if not scored_models:
            raise ValueError("No models available for multi-stage orchestration.")

        # Seed the first wave with the requested initial count (clamped by the
        # available pool and global parallelism guardrails).
        current_selection = [
            invocation
            for invocation, _score in scored_models[: self.config.initial_model_count]
        ]

        snapshots: List[RunSnapshot] = []
        progress_events: List[Dict[str, Any]] = []
        # Tracks which models have already been used, so that disagreement-driven
        # expansions can introduce fresh perspectives instead of recycling.
        used_model_ids: set[str] = set()

        async def emit_progress(
            message: str,
            *,
            progress: Optional[float] = None,
            total: Optional[float] = None,
        ) -> None:
            event: Dict[str, Any] = {"message": message}
            if progress is not None:
                event["progress"] = progress
            if total is not None:
                event["total"] = total
            progress_events.append(event)

            if context is not None:
                try:
                    if progress is not None:
                        await context.report_progress(progress, total, message)
                    await context.info(message)
                except Exception:
                    # Progress notifications should be best-effort; ignore transport errors.
                    pass

        for round_index in range(1, self.config.max_runs + 1):
            if not current_selection:
                break  # Safety: nothing left to run.

            capped_selection = current_selection[: self.config.max_parallel_models]
            used_model_ids.update(inv.model.model_id for inv in capped_selection)

            await emit_progress(
                f"Run {round_index}: launching {len(capped_selection)} invocations",
                progress=round_index - 1,
                total=float(self.config.max_runs),
            )

            # Execute the wide parallel pass for this round.
            results = await self._execute_wave(
                task,
                capped_selection,
                round_index,
                emit_progress,
            )
            agreement, agreement_level = self._measure_agreement(results)

            snapshot = RunSnapshot(
                run_index=round_index,
                model_ids=[
                    result.metadata.get("invocation_label", result.model_id)
                    for result in results
                ],
                results=results,
                agreement_level=agreement_level,
                average_similarity=agreement,
                notes={
                    "selection_size": len(capped_selection),
                    "unique_models_seen": len(used_model_ids),
                },
            )
            snapshots.append(snapshot)

            await emit_progress(
                f"Run {round_index} completed with {agreement_level.value} (avg similarity {agreement:.2f})",
                progress=round_index,
                total=float(self.config.max_runs),
            )

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
            task,
            final_model,
            snapshots,
        )

        await emit_progress(
            "Final synthesis completed",
            progress=float(self.config.max_runs),
            total=float(self.config.max_runs),
        )

        return final_synthesis, snapshots, progress_events

    # ------------------------------------------------------------------
    # Model selection utilities
    # ------------------------------------------------------------------

    def _build_invocations(
        self,
        models: Iterable[ModelInfo],
        candidate_models: Optional[Sequence[str]],
    ) -> List[ModelInvocation]:
        """Prepare the invocation roster using the configured default plan."""

        lookup: Dict[str, ModelInfo] = {model.model_id: model for model in models}
        invocations: List[ModelInvocation] = []

        # Prepare base plan (default + user-specified extras).
        plan_sequence: List[ModelPlan] = list(self.default_plan)

        if not plan_sequence:
            plan_sequence = [ModelPlan(model_id=model_id) for model_id in lookup.keys()]

        plan_ids_lower = {plan.model_id.lower() for plan in plan_sequence}

        for model_id in candidate_models or []:
            if model_id.lower() not in plan_ids_lower:
                plan_sequence.append(ModelPlan(model_id=model_id))
                plan_ids_lower.add(model_id.lower())

        for plan in plan_sequence:
            model = lookup.get(plan.model_id)
            if model is None:
                continue

            for idx in range(max(1, plan.count)):
                invocations.append(
                    ModelInvocation(
                        model=model,
                        plan=plan,
                        instance_index=idx,
                    )
                )

        return invocations

    def _score_invocations(
        self, invocations: Iterable[ModelInvocation]
    ) -> List[Tuple[ModelInvocation, float]]:
        """Score each invocation using provider prestige, context and cost."""

        scored: List[Tuple[ModelInvocation, float]] = []

        for invocation in invocations:
            model = invocation.model
            provider_score = self._provider_priority(model.provider)
            context_multiplier = math.log(max(model.context_length, 2048), 2) / 10.0
            cost_component = min(model.cost_per_token or 0.00002, 0.0005) * 100.0

            score = provider_score + context_multiplier + cost_component
            scored.append((invocation, score))

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
        invocations: Sequence[ModelInvocation],
        run_index: int,
        progress_callback: Callable[..., Awaitable[None]],
    ) -> List[ProcessingResult]:
        """
        Fire off a single wave of model calls concurrently.

        Each invocation maintains its own label so progress events can reference
        duplicate instances of the same base model (e.g. multiple GPT-5 runs).
        """

        async def invoke(invocation: ModelInvocation) -> ProcessingResult:
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
                    "orchestration_model": invocation.model.model_id,
                    "invocation_label": invocation.label,
                    "requested_options": invocation.plan.request_options,
                },
            )
            return await asyncio.wait_for(
                self.model_provider.process_task(
                    cloned_task,
                    invocation.model.model_id,
                ),
                timeout=self.config.timeout_seconds,
            )

        async_tasks: List[asyncio.Task[ProcessingResult]] = []
        for invocation in invocations:
            async_task = asyncio.create_task(invoke(invocation))
            setattr(async_task, "_orchestrator_invocation", invocation)
            async_tasks.append(async_task)

        results: List[ProcessingResult] = []

        pending: set[asyncio.Task[ProcessingResult]] = set(async_tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task_future in done:
                invocation = getattr(task_future, "_orchestrator_invocation")
                try:
                    result = await task_future
                    result.metadata["invocation_label"] = invocation.label
                    results.append(result)
                    await progress_callback(
                        f"Run {run_index}: {invocation.label} completed",
                    )
                except asyncio.TimeoutError:
                    await progress_callback(
                        f"Run {run_index}: {invocation.label} timed out after {self.config.timeout_seconds:.0f}s",
                    )
                except Exception as exc:  # pragma: no cover - defensive logging path
                    await progress_callback(
                        f"Run {run_index}: {invocation.label} failed ({exc})",
                    )

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
        scored_invocations: Sequence[Tuple[ModelInvocation, float]],
        used_model_ids: set[str],
        last_results: Sequence[ProcessingResult],
        average_similarity: float,
        agreement_level: AgreementLevel,
    ) -> List[ModelInvocation]:
        """
        Decide which model invocations should participate in the next run.

        The policy mirrors the narrative specification:
        - High agreement: halve the roster, keeping the highest-confidence answers.
        - Moderate agreement: keep roster size steady but rotate in a fresh voice.
        - Low/no agreement: expand with unused models to gather more perspectives.
        """

        invocation_lookup = {
            invocation.label: invocation for invocation, _ in scored_invocations
        }
        fallback_lookup: Dict[str, ModelInvocation] = {}
        for invocation, _ in scored_invocations:
            fallback_lookup.setdefault(invocation.model.model_id, invocation)

        def resolve_result(result: ProcessingResult) -> Optional[ModelInvocation]:
            label = result.metadata.get("invocation_label")
            if label and label in invocation_lookup:
                return invocation_lookup[label]
            return fallback_lookup.get(result.model_id)

        sorted_results = sorted(
            last_results,
            key=lambda res: res.confidence,
            reverse=True,
        )
        sorted_invocations = [resolve_result(res) for res in sorted_results]
        sorted_invocations = [inv for inv in sorted_invocations if inv is not None]

        if agreement_level in (AgreementLevel.HIGH_CONSENSUS, AgreementLevel.UNANIMOUS):
            keep_count = max(
                self.config.minimum_model_count,
                max(1, len(sorted_invocations) // 2),
            )
            return sorted_invocations[:keep_count] or sorted_invocations

        if agreement_level == AgreementLevel.MODERATE_CONSENSUS:
            next_wave = sorted_invocations.copy()
            replacement_candidates = [
                inv
                for inv, _ in scored_invocations
                if inv.model.model_id not in used_model_ids and inv not in next_wave
            ]
            if replacement_candidates and next_wave:
                next_wave[-1] = replacement_candidates[0]
                used_model_ids.add(replacement_candidates[0].model.model_id)
            return next_wave or sorted_invocations

        # Low or no consensus: recruit more voices if possible.
        expanded: List[ModelInvocation] = list(sorted_invocations)
        if not expanded:
            expanded = [inv for inv, _ in scored_invocations[: self.config.minimum_model_count]]

        for inv, _ in scored_invocations:
            if len(expanded) >= self.config.max_parallel_models:
                break
            if inv in expanded:
                continue
            if inv.model.model_id in used_model_ids:
                continue
            expanded.append(inv)
            used_model_ids.add(inv.model.model_id)

        return expanded[: self.config.max_parallel_models]

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    def _select_final_model(
        self,
        scored_invocations: Sequence[Tuple[ModelInvocation, float]],
        snapshots: Sequence[RunSnapshot],
    ) -> ModelInfo:
        """
        Choose the highest-priority model that actually participated (or is available).

        We give preference to models that have already produced content in earlier
        runs with high confidence, falling back to the static priority ladder.
        """
        participating_ids = {
            result.model_id for snapshot in snapshots for result in snapshot.results
        }

        priority_chain: List[str] = list(self.model_priority)
        if self.final_synthesis_model_id:
            priority_chain = [
                self.final_synthesis_model_id,
                *[pid for pid in priority_chain if pid != self.final_synthesis_model_id],
            ]

        for preferred_id in priority_chain:
            for invocation, _ in scored_invocations:
                if invocation.model.model_id == preferred_id:
                    if not participating_ids or preferred_id in participating_ids:
                        return invocation.model

        # Fallback: pick the highest scoring model from the scored list.
        return scored_invocations[0][0].model

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
                    f"- {result.metadata.get('invocation_label', result.model_id)}: "
                    f"confidence {result.confidence:.2f} → {truncated}"
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
            synthesis_task,
            final_model.model_id,
            **self.final_synthesis_options,
        )

        return FinalSynthesis(
            model_id=final_model.model_id,
            result=final_result,
            prompt_tokens=len(synthesis_prompt.split()),
            summary="\n".join(summary_lines),
        )
