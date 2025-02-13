from __future__ import annotations

import asyncio
import itertools
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Iterable, Optional, Type, Union

import instructor
from instructor.exceptions import InstructorRetryException
from loguru import logger
from openai import (APIConnectionError, AsyncAzureOpenAI, AsyncOpenAI,
                    AzureOpenAI, BadRequestError)
from pydantic import BaseModel, Field, ValidationError
from rich import print
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

install()

custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "danger": "bold red"})
console = Console(theme=custom_theme)


class Chain(BaseModel, metaclass=ABCMeta):

    input_schema: ClassVar[Type[Any]]
    output_schema: ClassVar[Type[Any]]

    @classmethod
    @abstractmethod
    def make_input_text(cls, *, input: Any) -> str:
        pass

    @classmethod
    def make_inputs(cls, *, input_objects: list[Type[BaseModel]]) -> list[str]:
        for input_object in input_objects:
            check = isinstance(input_object, cls.input_schema)
            if check is False:
                print("input object")
                print(input_object)
                print("input schema")
                print(cls.input_schema)
                raise ValueError(f"input object is not of type {cls.input_schema()}")
        prompts = [
            cls.make_input_text(input=input_object) for input_object in input_objects
        ]
        return prompts

    @classmethod
    def make_client(
        cls, llm_name: str, timeout: Union[int, None]
    ) -> Union[AsyncAzureOpenAI, AzureOpenAI]:
        endpoint = endpoints[llm_name]

        if llm_name == "o3-mini":
            api_version = "2024-12-01-preview"
            api_key = "PLACEHOLDER"
        else:
            raise ValueError(f"llm_name {llm_name} not implemented")

        deployment_client = instructor.from_openai(
            AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_deployment=llm_name,
                api_key=api_key,
                timeout=timeout,
            )
        )
        return deployment_client  # type: ignore

    @classmethod
    async def coroutine(
        cls,
        *,
        client,
        llm_name: str,
        prompt: str,
        max_retries: int,
        max_tokens: int,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        try:
            if reasoning_effort is not None:
                result = await client.chat.completions.create(  # type: ignore
                    model=llm_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_model=cls.output_schema,
                    max_retries=max_retries,
                    # max_tokens=max_tokens,
                )
            else:
                # o3-mini
                result = await client.chat.completions.create(  # type: ignore
                    model=llm_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_model=cls.output_schema,
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                )
        except InstructorRetryException as e:
            print(prompt)
            logger.warning(f"Retry Exception: {e}")
            return e
        except BadRequestError as e:
            print(prompt)
            logger.warning(f"Risky Content: {e}")
            return e
        except ValidationError as e:
            print(prompt)
            logger.warning(f"Validation error: {e}")
            return e
        except Exception as e:
            print(prompt)
            logger.error(f"Unknown Exception: {e}")
            # website.chain:coroutine:136 - Unknown Exception: Connection error.
            return e
        return result

    @classmethod
    async def batch_predict(
        cls,
        *,
        max_tokens: int,
        size: int,
        llm_name: str,
        timeout: Union[int, None],
        max_retries: int,
        input_objects: Iterable[Any],
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> list[Any]:
        responses = []
        batch_size = size
        name = cls.__name__
        try:
            input_objects_batches = list(itertools.batched(input_objects, batch_size))
        except Exception as e:
            print(input_objects)
            print(batch_size)
            print(e)
            breakpoint()
        # new client for each batch
        client = cls.make_client(llm_name, sync=False, timeout=timeout)
        for idx, input_objects_batch in enumerate(input_objects_batches):

            console.print(
                f"{name} idx {idx}/{len(input_objects_batches)} - batch size: {batch_size}",
                style="info",
            )

            prompts = cls.make_inputs(input_objects=input_objects_batch, **kwargs)
            tasks = []
            for prompt in prompts:
                # print(prompt)
                task = asyncio.create_task(
                    cls.coroutine(
                        client=client,
                        llm_name=llm_name,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        reasoning_effort=reasoning_effort,
                    )
                )
                tasks.append(task)

            try:
                responses.extend(await asyncio.gather(*tasks))
            except APIConnectionError as e:
                logger.warning(f"You batch {size} is too big? ...retrying {e}")
                raise Exception(f"error in batch predict: {e}")
            except Exception as e:
                # TOO MANY OPEN FILES
                logger.warning(f"error in batch predict: {e}")
        errors = [r for r in responses if isinstance(r, Exception)]
        N = len(errors)
        if N > 0:
            logger.warning(f"batch run for {name}: {N} errors")
        # client.close()
        return responses
