import asyncio
import collections.abc

import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.model
import inspect_ai.solver


def refusal_output(
    *, category: str = "cyber", explanation: str = "declined by classifier"
) -> inspect_ai.model.ModelOutput:
    return inspect_ai.model.ModelOutput.from_content(
        model="mockllm/model",
        content="I can't help with that.",
        stop_reason="content_filter",
        stop_details=inspect_ai.model.StopDetails(
            type="refusal", category=category, explanation=explanation
        ),
    )


def normal_output(content: str = "Sure, here you go.") -> inspect_ai.model.ModelOutput:
    return inspect_ai.model.ModelOutput.from_content(
        model="mockllm/model", content=content, stop_reason="stop"
    )


def run_eval_with_outputs(
    outputs: collections.abc.Sequence[inspect_ai.model.ModelOutput],
    *,
    generate_calls: int = 1,
) -> inspect_ai.log.EvalLog:
    """Run a one-sample eval whose solver calls generate `generate_calls` times."""

    @inspect_ai.solver.solver
    def multi_generate() -> inspect_ai.solver.Solver:
        async def solve(
            state: inspect_ai.solver.TaskState,
            generate: inspect_ai.solver.Generate,
        ) -> inspect_ai.solver.TaskState:
            for _ in range(generate_calls):
                state = await generate(state)
                # Sample events reach hooks via a background emitter task that
                # is scheduled cooperatively and lags well behind the near-
                # instant mockllm solver. This brief park lets the emitter
                # drain the queue, run our hook, and land its task-group cancel
                # while the solver is still awaiting -- so a refusal breach is
                # observed and terminated deterministically. Real models supply
                # this yield time naturally via network I/O.
                await asyncio.sleep(0.05)
            return state

        return solve

    model = inspect_ai.model.get_model("mockllm/model", custom_outputs=list(outputs))
    task = inspect_ai.Task(
        dataset=[inspect_ai.dataset.Sample(input="hello", target="hi")],
        solver=multi_generate(),
    )
    # inspect_ai.eval's own signature leaks Unknown (its `scanner` param), which
    # strict mode flags as a partially-unknown member. The type genuinely lives
    # in inspect_ai and cannot be narrowed here, so ignore this one rule only.
    logs = inspect_ai.eval(  # pyright: ignore[reportUnknownMemberType]
        task, model=model, display="none"
    )
    return logs[0]
