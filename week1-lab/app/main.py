import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# Import Datadog Tracer
from ddtrace import tracer

MODEL_PATH = "/tmp/simple_model.onnx"

# Dedicated thread pool capped to available cores
executor = ThreadPoolExecutor(max_workers=4)

def ensure_dummy_onnx_model():
    if os.path.exists(MODEL_PATH):
        return
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    two_tensor = helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])
    node_def = helper.make_node('Mul', inputs=['input', 'two'], outputs=['output'])

    graph_def = helper.make_graph([node_def], 'simple-mul', [X], [Y], initializer=[two_tensor])
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-demo',
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)]
    )

    with open(MODEL_PATH, "wb") as f:
        f.write(model_def.SerializeToString())


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dummy_onnx_model()
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    app.state.onnx_session = session
    
    # Warm-up pass
    input_name = session.get_inputs()[0].name
    dummy_data = np.ones((1, 4), dtype=np.float32)
    session.run(None, {input_name: dummy_data})
    
    yield
    
    app.state.onnx_session = None
    executor.shutdown(wait=False)


app = FastAPI(title="AI Inference Service", lifespan=lifespan)


class InferencePayload(BaseModel):
    inputs: list[float]


def run_onnx_inference(session: ort.InferenceSession, data: np.ndarray):
    """Synchronous CPU-bound execution function offloaded to thread pool."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: data})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "onnx_session") and app.state.onnx_session is not None}


@app.post("/predict")
async def predict(payload: InferencePayload, request: Request):
    session: ort.InferenceSession = getattr(request.app.state, "onnx_session", None)
    if not session:
        raise HTTPException(status_code=500, detail="ONNX model session not loaded.")

    try:
        loop = asyncio.get_running_loop()
        data = np.array([payload.inputs], dtype=np.float32)
        start_time = time.perf_counter()

        # Wrap threadpool model execution in a Datadog custom APM span
        with tracer.trace("onnx.inference", resource="simple_model_predict") as span:
            span.set_tag("model.name", "simple_mul")
            span.set_tag("model.batch_size", len(payload.inputs))
            
            results = await loop.run_in_executor(executor, run_onnx_inference, session, data)
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "predictions": results[0].tolist(),
            "latency_ms": round(latency_ms, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))