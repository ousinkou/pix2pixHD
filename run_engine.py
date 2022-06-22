import os
import sys
from random import randint
import numpy as np

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import tensorrt as trt
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)
import common
trt.init_libnvinfer_plugins(None, "")

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path="", use_fp16=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            if use_fp16:
                config.set_flag(trt.BuilderFlag.FP16) #FP16
            else:
                config.set_flag(trt.BuilderFlag.TF32)

            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            #output = network.get_output(0)
            #network.get_input(0).shape = [1, 1, 800, 800]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def run_onnx(onnx_file, data_type, bs, inp):
    engine_file_path = os.path.splitext(onnx_file)[0]+".engine"
    with get_engine(onnx_file, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = inp
        trt_outputs = common.do_inference_v2(context, bindings=bindings,
                                             inputs=inputs, outputs=outputs,
                                             stream=stream)

    return trt_outputs



