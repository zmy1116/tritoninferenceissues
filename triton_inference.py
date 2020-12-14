"""
Run triton server

Usage:
    triton_inference.py <model_name> <inputs_path> <repeat_num> <output_path>

Options:
    <model_name>                model name
    <inputs_path>               input path
    <repeat_num>                number of times repeat computation
    <output_path>               output stored path
    -h --help                   Show this screen
"""

import pickle
import logging
import numpy as np
from docopt import docopt
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype


class TritonGrpcModelClient(object):

    def __init__(self, url, model_name):

        self.triton_client = grpcclient.InferenceServerClient(url)
        self.model_name = model_name
        assert self.triton_client.is_server_ready()
        assert self.triton_client.is_model_ready(model_name)
        self.model_metadata = self.triton_client.get_model_metadata(model_name)
        self.model_config = self.triton_client.get_model_config(model_name)
        self.inputs_specs, self.outputs_specs, self.max_batch_size = self.parse_model_grpc()

    def parse_model_grpc(self):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        model_metadata = self.model_metadata
        model_config = self.model_config.config
        inputs_specs = model_metadata.inputs
        outputs_specs = model_metadata.outputs
        max_batch_size = model_config.max_batch_size

        return inputs_specs, outputs_specs, max_batch_size

    def inputs_outputs_generator(self, raw_inputs):
        """
        Generate inputs and outptus blob for triton client inference
        :param raw_inputs: list of raw numpy inputs
        :return: inputs outputs data
        """
        inputs = []
        for input_specs, raw_input in zip(self.inputs_specs, raw_inputs):
            # parse data type
            raw_input = raw_input.astype(triton_to_np_dtype(input_specs.datatype))
            infer_input = grpcclient.InferInput(input_specs.name, raw_input.shape, input_specs.datatype)
            infer_input.set_data_from_numpy(raw_input)
            inputs.append(infer_input)

        outputs = []
        for output_specs in self.outputs_specs:
            outputs.append(grpcclient.InferRequestedOutput(output_specs.name, class_count=0))
        return inputs, outputs

    def do_inference(self, inputs_list):
        """
        do inference
        :param inputs_list: list of raw numpy inputs
        :return:
        """
        # TODO deal with case when batch size > max batch size

        while True:
            model_inputs, model_outputs = self.inputs_outputs_generator(inputs_list)
            results = self.triton_client.infer(self.model_name, model_inputs, outputs=model_outputs)
            outputs = self.results_parsing(results)
            if self.verify_result(outputs):
                return outputs

    def results_parsing(self, results):
        outputs = []
        for output_specs in self.outputs_specs:
            outputs.append(
                results.as_numpy(output_specs.name)
            )
        return outputs

    @staticmethod
    def verify_result(result):
        for r in result:
            if np.isnan(r).any():
                return False
        return True

    def repeatedly_evaluate_inputs(self, inputs_list, repeat_num):

        outputs = []
        for _ in range(repeat_num):
            outputs.append(self.do_inference(inputs_list)[0])

        outputs = np.concatenate(outputs)
        return outputs


if __name__ == '__main__':
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    model_name = arguments['<model_name>']
    inputs_path = arguments['<inputs_path>']
    output_path = arguments['<output_path>']
    repeat_num = int(arguments['<repeat_num>'])

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)s] [%(name)s] %(levelname)s: %(message)s'
    )
    logging.info('Load Model GRPC client')
    inference_module = TritonGrpcModelClient('localhost:8001', model_name)

    logging.info('Load input data, do inference')
    inputs_data = pickle.load(open(inputs_path, 'rb'))
    outputs = inference_module.repeatedly_evaluate_inputs([inputs_data], repeat_num)

    logging.info('Store outputs')
    pickle.dump(outputs, open(output_path, 'wb'))
