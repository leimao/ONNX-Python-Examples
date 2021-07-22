import numpy as np
import onnxruntime as rt


def main() -> None:

    model_file_path = "convnet.onnx"
    sess = rt.InferenceSession(model_file_path)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    batch_size = 1
    dummy_input = np.random.random(
        (batch_size, *input_shape[1:])).astype(np.float32)
    prediction = sess.run(None, {input_name: dummy_input})[0]


if __name__ == "__main__":

    main()
