import onnxruntime
def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

model = init_onnx_model(model_path='./model.onnx')