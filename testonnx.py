import torch
import onnxruntime
from pprint import pprint

_, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts,
 get_speech_ts_adaptive,
 _, read_audio,
 _, _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs[0]
    
model = init_onnx_model(f'{files_dir}/model.onnx')
wav = read_audio(f'{files_dir}/en.wav')

# get speech timestamps from full audio file

# classic way
speech_timestamps = get_speech_ts(wav, model, num_steps=4, run_function=validate_onnx) 
pprint(speech_timestamps)

# adaptive way
speech_timestamps = get_speech_ts(wav, model, run_function=validate_onnx) 
pprint(speech_timestamps)
