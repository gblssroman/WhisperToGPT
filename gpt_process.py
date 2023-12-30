import g4f
import asyncio

g4f.debug.logging = False
g4f.debug.version_check = False

providers = [
    g4f.Provider.FakeGpt,
]

#todo: provider choice

async def call_provider(
        provider: g4f.Provider.BaseProvider, message: str, lang: str, prev_msg: str, status: dict
):
    """Calling the GPT-3.5 providers"""
    if status['received']:
        return

    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.gpt_35_turbo,
            messages=[{
                "role": "user", "content":
                    f" {'Previous message:' + prev_msg if len(prev_msg) > 0 else prev_msg}."
                    f" Current question: {message}. Answer on language: '{lang}' clearly."
            }],
            provider=provider,
            stream=False,
            timeout=5,
        )

        if len(response) > 1 and not status['received']:
            status['received'] = True
            print(f"\nResponse:\n{response}\n")
        # print(f"{provider.__name__}:", response)

    except Exception as e:
        status['exception'] = e
        pass  # no need of printing responses containing errors here


async def get_response(message: str, lang: str, prev_msg: str):
    """Async structure of sending prompts"""
    status = {'received': False, 'exception': ''}
    calls = [
        call_provider(provider, message, lang, prev_msg, status) for provider in providers
    ]
    await asyncio.gather(*calls)

    if not status['received']:
        print(f"\nRequests failed! Last error: {status['exception']}")


async def call_main(message: str, lang: str, prev_msg: str):
    """"""
    await get_response(message, lang, prev_msg)
