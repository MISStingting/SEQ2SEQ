from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
import tensorflow as tf
import numpy as np


class ServingClient(object):
    def __init__(self,
                 model_name,
                 host,
                 port,
                 timeout=30):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.timeout = timeout
        channel = implementations.insecure_channel(self.host, self.port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def request(self, address):
        addr = address.split(" ")
        input = np.asarray([addr], np.str)
        shape = input.shape
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.inputs["input"].CopyFrom(tf.make_tensor_proto(input, shape=[shape[0], shape[1]]))
        request.inputs["input_length"].CopyFrom(tf.make_tensor_proto(shape[1], shape=[1, ]))
        future = self.stub.Predict.future(request, self.timeout)
        result = future.result()
        # sync requests
        # result_future = stub.Predict(request, 30.)
        # For async requests
        # result_future = stub.Predict.future(request, 10.)
        # Do some work...
        # result_future = result_future.result()
        labels = tf.make_ndarray(result.outputs["predict_labels"])[0]
        labels = [e.decode("utf8") for e in labels]
        print("\n")
        return labels


if __name__ == '__main__':
    ocr = r"E:\LTTProject\SEQ2SEQ\data\dev.ocr"
    std = r"E:\LTTProject\SEQ2SEQ\data\dev.std"
    serve_client = ServingClient(model_name="seqAddress", host="10.100.3.200", port=8090, timeout=30)

    with open(file=ocr, mode="rt", encoding="utf8", buffering=8192) as f, \
            open(file=std, mode="rt", encoding="utf8", buffering=8192) as fin:
        total, differ, same = 0, 0, 0
        for line in f:
            ds_ocr = line.strip("\n")

            ds_std = fin.readline().strip("\n")
            ds_res = serve_client.request(ds_ocr)[:len(ds_ocr.split(" "))]
            ds_res = " ".join(ds_res)
            total += 1
            if ds_std == ds_res:
                same += 1
            else:
                differ += 1
            print("ocr:", ds_ocr)
            print("std:", ds_std)
            print("res:", ds_res)
        print("same %:", (same / total))
