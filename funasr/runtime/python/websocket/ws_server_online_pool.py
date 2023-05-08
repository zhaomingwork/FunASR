import asyncio
import json
import websockets
import time
from queue import Queue
import threading
import logging
import tracemalloc
import numpy as np
import traceback
from parse_args import args
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from funasr_onnx.utils.frontend import load_bytes
from concurrent.futures import ThreadPoolExecutor

tracemalloc.start()

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

websocket_users = set()

print("model loading,num of threads=", args.num_threads)
inference_pipeline_asr_online = []
mutex = threading.Lock()
for i in range(args.num_threads):
    infer_online = pipeline(task=Tasks.auto_speech_recognition,
                            model=args.asr_model_online,
                            model_revision='v1.0.4')
    print("total num=", args.num_threads, "loaded model ", i)
    infer_online.isused = False
    inference_pipeline_asr_online.append(infer_online)


async def print_avail_server():
    avail_num = 0
    for i in range(len(inference_pipeline_asr_online)):
        if inference_pipeline_asr_online[i].isused == False:
            avail_num = avail_num + 1
    print("available threads number=", avail_num, flush=True)


async def get_avail_infer():
    await print_avail_server()
    mutex.acquire()
    for i in range(len(inference_pipeline_asr_online)):
        if inference_pipeline_asr_online[i].isused == False:
            inference_pipeline_asr_online[i].isused = True
            mutex.release()
            return inference_pipeline_asr_online[i]
    mutex.release()
    return None


print("model loaded, available threads=", len(inference_pipeline_asr_online))

threads_pool = ThreadPoolExecutor(max_workers=args.num_threads, )

threadid = 0


async def ws_serve(websocket, path):
    frames_online = []
    global websocket_users, threadid
    websocket.send_msg = Queue()
    websocket_users.add(websocket)
    websocket.param_dict_asr_online = {"cache": dict()}
    websocket.speek_online = Queue()
    avail_infer = await get_avail_infer()

    if not avail_infer is None:
        websocket.infer = avail_infer
    else:

        message = json.dumps({"mode": "online", "text": ["srv not available"]})
        await websocket.send(message)
        await websocket.close()

    try:
        async for message in websocket:
            message = json.loads(message)
            is_finished = message["is_finished"]
            if not is_finished:
                audio = bytes(message['audio'], 'ISO-8859-1')
                is_speaking = message["is_speaking"]
                websocket.param_dict_asr_online["is_final"] = not is_speaking
                websocket.param_dict_asr_online["chunk_size"] = message[
                    "chunk_size"]
                frames_online.append(audio)
                if len(frames_online
                       ) % message["chunk_interval"] == 0 or not is_speaking:
                    audio_in = b"".join(frames_online)
                    frames_online = []
                    loop = asyncio.get_running_loop()
                    msg = await loop.run_in_executor(
                        threads_pool,
                        asr_online,
                        websocket,
                        audio_in,
                    )
                    if len(msg) > 0:
                        await websocket.send(msg)

    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users)  # 链接断开
        mutex.acquire()
        websocket.infer.isused = False
        mutex.release()
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")  # 无效状态
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
        exit(0)


def asr_online(websocket, audio_in):  # ASR推理
    if len(audio_in) > 0:
        audio_in = load_bytes(audio_in)
        rec_result = websocket.infer(
            audio_in=audio_in, param_dict=websocket.param_dict_asr_online)
        if websocket.param_dict_asr_online["is_final"]:
            websocket.param_dict_asr_online["cache"] = dict()
        if "text" in rec_result:
            if rec_result["text"] != "sil" and rec_result[
                    "text"] != "waiting_for_more_voice":
                message = json.dumps({
                    "mode": "online",
                    "text": rec_result["text"]
                })
                return message
        return ""


start_server = websockets.serve(ws_serve,
                                args.host,
                                args.port,
                                subprotocols=["binary"],
                                ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
