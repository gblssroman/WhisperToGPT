import torch.cuda
import asyncio
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import argparse
import os
from datetime import datetime
import numpy as np
import gpt_process

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='medium', choices=['tiny', 'small', 'medium', 'large'],
                    help='choose the model to use for transcribing')
parser.add_argument('--lang', type=str, default='all', choices=['all', 'en', 'ru'],
                    help='choose the language: en for English; ru for Russian; all for other languages (default)')
parser.add_argument('--cuda', type=int, default=1, choices=[0, 1],
                    help='use CUDA/ROCm: 0 for CPU; 1 for GPU (default)')
parser.add_argument('--sound_device', type=int, default=6,
                    help='select audio device (mic): integer expected; to query devices pass --query-devices;'
                         'default - 6')
parser.add_argument('--channels', type=int, default=2,
                    help='select channels of audio device (mic): integer expected; default - 2')
parser.add_argument('--query_devices', type=str, default='false', choices=['true', 'false'],
                    help='pass true to query all sound devices available')
parser.add_argument('--save_audio', type=str, default='true', choices=['true', 'false'],
                    help='pass false to automatically delete all .wav outputs; default - true')
parser.add_argument('--timer', type=int, default=5, choices=[5, 8, 10, 15, 20],
                    help='set the time of one audio chunk; default - 5')
parser.add_argument('--provider', type=str, default='FakeGpt',
                    choices=['FakeGpt', 'GeekGpt', 'Liaobots', 'Raycast'],
                    help='set the GPT provider; default - FakeGpt')

args = parser.parse_args()
output_dir = f"voice_output/{datetime.today().strftime('%Y-%m-%d')}"
fs, chunk_dur, chunk_count, prev_msg = 44100, args.timer, 0, ''

if bool(args.cuda) and torch.cuda.is_available():
    device = 'cuda'
    print(f"Running on {torch.cuda.get_device_name(0)}...")
else:
    device = 'cpu'
    print("\nUsing CPU...")

model = whisper.load_model(f"{args.model}.en" if (args.lang == 'en' and args.model != 'large') else args.model,
                           device=device)
print("\nModel ready... Voice recognition started!")

async def trasncribe_audio():
    """Transcribe it!"""
    global prev_msg, chunk_count
    result = model.transcribe(
        audio=f"{output_dir}/{chunk_count}.wav",
        language='russian' if args.lang == 'ru' else None
    )
    chunk_count += 1
    if args.save_audio != 'true':
        chunk_count -= 1
        os.remove(f"{output_dir}/{chunk_count}.wav")

    print("-" * 20 + f"\nYou said: {result['text']},\nLanguage detected: {result['language']}\n" + "-" * 20)
    if result['language'] != 'nn':
        await gpt_process.call_main(result['text'], result['language'], prev_msg)
        prev_msg = result['text']
    else:
        print("It is silent! Skipping chunk...")


async def audio_rec(chunk_count):
    """Recording process and storing .wav"""
    current_frame = sd.rec(frames=fs * chunk_dur, samplerate=fs, channels=args.channels, dtype=np.float32)
    sd.wait()
    wav.write(f"{output_dir}/{chunk_count}.wav", rate=fs, data=current_frame)

async def main():
    """Main WhispCopilot async initialization"""
    os.makedirs(output_dir, exist_ok=True)

    while True:
        rec_task = asyncio.create_task(audio_rec(chunk_count))
        transcribe_task = asyncio.create_task(trasncribe_audio())

        await rec_task
        await transcribe_task

if __name__ == '__main__':
    if args.query_devices == 'true':
        print(sd.query_devices())
    else:
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"\nEncountered an error... \n{e.__str__()}\n")
