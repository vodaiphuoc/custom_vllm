import asyncio
from contextlib import ExitStack
from unittest.mock import MagicMock
import pytest

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from vllm.inputs import PromptType, StreamingInput
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

from collections.abc import AsyncGenerator



TEXT_ENGINE_ARGS = AsyncEngineArgs(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=True,
)

Standard_SP = SamplingParams(
    max_tokens=2,
    ignore_eos=True,
    output_kind=RequestOutputKind.DELTA,
    temperature=1.0,
    seed=33,
    n=1,
    prompt_logprobs=None,
)


FULL_QUESTION = "Hello my name is Robert and what is president of the US"


async def STREAM_PROMPT():
    for word in FULL_QUESTION.split():
        yield StreamingInput(prompt=word, sampling_params=Standard_SP)



async def generate(
    engine: AsyncLLM,
    request_id: str,
    prompt: AsyncGenerator[StreamingInput],
    sampling_params = Standard_SP,
    cancel_after: int | None = None,
) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)

    count = 0
    
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        print(out.outputs)
        num_tokens = sum(len(output.token_ids) for output in out.outputs)
        if sampling_params.output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        if cancel_after is not None and count >= cancel_after:
            return count, request_id

        await asyncio.sleep(0.0)

    return count, request_id


@pytest.mark.parametrize(
    "engine_args",
    [(TEXT_ENGINE_ARGS)],
)
@pytest.mark.asyncio
async def test_load(
    engine_args: AsyncEngineArgs,
):
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        tasks = asyncio.create_task(
                    generate(
                        engine= engine, 
                        request_id = request_id, 
                        prompt = STREAM_PROMPT,
                    )
        )

        # Confirm that we got all the EXPECTED tokens from the requests.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        
        for task in done:
            num_generated_tokens, request_id = await task
            

