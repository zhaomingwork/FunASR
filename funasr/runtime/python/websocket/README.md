# Service with websocket-python

This is a demo using funasr pipeline with websocket python-api. It supports the offline, online, offline/online-2pass unifying speech recognition. 

## For the Server

### Install the modelscope and funasr

```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

### Install the requirements for server

```shell
cd funasr/runtime/python/websocket
pip install -r requirements_server.txt
```

### Start server

##### API-reference
```shell
python wss_srv_asr.py \
--port [port id] \
--asr_model [asr model_name] \
--asr_model_online [asr model_name] \
--punc_model [punc model_name] \
--ngpu [0 or 1] \
--ncpu [1 or 4] \
--certfile [path of certfile for ssl] \
--keyfile [path of keyfile for ssl] 
```
##### Usage examples
```shell
python wss_srv_asr.py --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  --asr_model_online "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
```

## For the client

Install the requirements for client
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/websocket
pip install -r requirements_client.txt
```

### Start client
#### API-reference
```shell
python wss_client_asr.py \
--host [ip_address] \
--port [port id] \
--chunk_size ["5,10,5"=600ms, "8,8,4"=480ms] \
--chunk_interval [duration of send chunk_size/chunk_interval] \
--words_max_print [max number of words to print] \
--audio_in [if set, loadding from wav.scp, else recording from mircrophone] \
--output_dir [if set, write the results to output_dir] \
--send_without_sleep [only set for offline] \
--ssl [1 for wss connect, 0 for ws, default is 1] \
--mode [`online` for streaming asr, `offline` for non-streaming, `2pass` for unifying streaming and non-streaming asr] \
```

#### Usage examples
##### ASR offline client
Recording from mircrophone
```shell
# --chunk_interval, "10": 600/10=60ms, "5"=600/5=120ms, "20": 600/12=30ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode offline --chunk_interval 10 --words_max_print 100
```
Loadding from wav.scp(kaldi style)
```shell
# --chunk_interval, "10": 600/10=60ms, "5"=600/5=120ms, "20": 600/12=30ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode offline --chunk_interval 10 --words_max_print 100 --audio_in "./data/wav.scp" --send_without_sleep --output_dir "./results"
```

##### ASR streaming client
Recording from mircrophone
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode online --chunk_size "5,10,5" --words_max_print 100
```
Loadding from wav.scp(kaldi style)
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode online --chunk_size "5,10,5" --audio_in "./data/wav.scp" --output_dir "./results"
```

##### ASR offline/online 2pass client
Recording from mircrophone
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode 2pass --chunk_size "8,8,4"
```
Loadding from wav.scp(kaldi style)
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python wss_client_asr.py --host "0.0.0.0" --port 10095 --mode 2pass --chunk_size "8,8,4" --audio_in "./data/wav.scp" --output_dir "./results"
```
## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/fix_bug_for_python_websocket) for contributing the websocket service.
3. We acknowledge [cgisky1980](https://github.com/cgisky1980/FunASR) for contributing the websocket service of offline model.
