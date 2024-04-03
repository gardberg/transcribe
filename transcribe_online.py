from whisper_online import *
from sounddevice import CallbackFlags, InputStream
import sounddevice as sd
import os
import queue
import argparse

TARGET_SAMPLERATE = 16000
SLEEP_S = 60

QUEUE_THRESHOLD = int(TARGET_SAMPLERATE * 2) # number of audio samples to store in queue before we transcribe

BLOCKSIZE = QUEUE_THRESHOLD # size of individual audio blocks created by input stream, passed to queue

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribe audio from microphone')
    parser.add_argument('--device', type=int, default=2, help='Device number to use for input')
    parser.add_argument('--lang', type=str, default='en', help='Source language')
    parser.add_argument('--model', type=str, default='tiny', help='Model to use')
    parser.add_argument('--compute', type=str, default='int8', help='Compute type')
    parser.add_argument('--devices', action='store_true', default=False, help='List available input devices')
    return parser.parse_args()

args = parse_args()

if args.devices:
    print(sd.query_devices())
    exit()

DEVICE_NO = os.getenv("DEVICE_NO", args.device)

device_info = sd.query_devices(DEVICE_NO, 'input')

src_lan = args.lang

model = FasterWhisperASR(src_lan, args.model, device="cpu", compute_type=args.compute)

model.use_vad()

tokenizer = create_tokenizer(src_lan)
model_online = OnlineASRProcessor(model, tokenizer=tokenizer)

chunk_queue = queue.Queue()
queue_size = 0

def audio_callback(indata: np.ndarray, frames: int, time, status: CallbackFlags):
    if status: print(status, file=sys.stderr)

    # do other processing here    

    global queue_size
    chunk_queue.put(indata)
    queue_size += len(indata)
    if queue_size >= QUEUE_THRESHOLD:
        audio = np.concatenate(list(chunk_queue.queue))

        model_online.insert_audio_chunk(audio)
        o = model_online.process_iter() # outputs (start_time, end_time, text), if no transcription is done: (None, None, '')
        o = model_online.finish()
        model_online.init()
        if o[0] is not None: print(o[2][1:])

        chunk_queue.queue.clear()
        queue_size = 0
    else:
        pass

stream = InputStream(
                device=DEVICE_NO, channels=1, samplerate=TARGET_SAMPLERATE,
                dtype=np.float32, callback=audio_callback,
                blocksize=BLOCKSIZE)

with stream:
    print(f"Using device: {device_info['name']}")
    print(f"Transcribing... {(src_lan)}")
    print()

    input("Press Enter to stop...\n")
    print()
