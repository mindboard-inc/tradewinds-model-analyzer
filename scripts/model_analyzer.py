import os
import sys
from typing import Dict, Any, List, Optional, Union
import tensorflow as tf
import keras
import h5py
import torch
import onnx
from functools import reduce
import operator
import numpy as np


class SavedModelAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_format = self.detect_model_format()
        self.model = None
        self.model_info = {
            'format': self.model_format,
            'layers': [],
            'total_params': 0,
            'input_shape': None,
            'output_shape': None,
            'compute_requirements': None,
            'memory_usage': None,
            'bottlenecks': []
        }

    def detect_model_format(self) -> str:
        _, ext = os.path.splitext(self.model_path)
        if ext == '.pb' or os.path.isdir(self.model_path):
            return 'tensorflow'
        elif ext in ['.h5', '.keras']:
            return 'keras'
        elif ext in ['.pt', '.pth']:
            return 'pytorch'
        elif ext == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def load_model(self):
        if self.model_format == 'tensorflow':
            self.model = tf.saved_model.load(self.model_path)
        elif self.model_format == 'keras':
            self.model = keras.models.load_model(self.model_path)
        elif self.model_format == 'pytorch':
            self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
        elif self.model_format == 'onnx':
            self.model = onnx.load(self.model_path)

    def analyze_model(self):
        if self.model_format == 'tensorflow':
            self.analyze_tensorflow_model()
        elif self.model_format == 'keras':
            self.analyze_keras_model()
        elif self.model_format == 'pytorch':
            self.analyze_pytorch_model()
        elif self.model_format == 'onnx':
            self.analyze_onnx_model()

    def analyze_keras_model(self):
        self.model_info['total_params'] = self.model.count_params()
        # Ensure all layers are built
        if not self.model.built:
            self.model.build()

        self.model_info['input_shape'] = [self.get_shape(layer) for layer in self.model.inputs]
        self.model_info['output_shape'] = [self.get_shape(layer) for layer in self.model.outputs]
        
        total_flops = 0
        total_memory = 0
        
        for layer in self.model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'input_shape': self.get_shape(layer.input),
                'output_shape': self.get_shape(layer.output),
                'params': layer.count_params()
            }

            # Handle InputLayer separately
            if layer_info['type'] == 'InputLayer':
                layer_info['input_shape'] = 'N/A'
                layer_info['output_shape'] = self.get_shape(layer.output)
                layer_info['params'] = 0
                layer_info['flops'] = 0
                layer_info['memory'] = 0
            else:
                flops, memory = self.estimate_layer_compute_memory(layer)
                layer_info['flops'] = flops
                layer_info['memory'] = memory
                total_flops += flops
                total_memory += memory
            
            self.model_info['layers'].append(layer_info)
        
        self.model_info['compute_requirements'] = f"{total_flops / 1e9:.2f} GFLOPs"
        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def get_shape(self, tensor: Union[tf.Tensor, keras.layers.Layer]) -> Union[tuple, List[tuple]]:
        if tensor is None:
            return 'Unknown'
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            if isinstance(shape, tf.TensorShape):
                return tuple(shape.as_list())
            return tuple(shape)
        elif hasattr(tensor, 'get_shape'):
            return tuple(tensor.get_shape().as_list())
        else:
            return 'Unknown'

    def analyze_tensorflow_model(self):
        concrete_func = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.model_info['input_shape'] = [t.shape.as_list() for t in concrete_func.inputs if not t.name.startswith('keras_learning_phase')]
        self.model_info['output_shape'] = [t.shape.as_list() for t in concrete_func.outputs]
        
        total_flops = 0
        total_memory = 0
        
        for node in concrete_func.graph.as_graph_def().node:
            layer_info = {
                'name': node.name,
                'type': node.op,
                'input': list(node.input),
                'output': [node.name]
            }
            flops, memory = self.estimate_tf_node_compute_memory(node, concrete_func.graph)
            layer_info['flops'] = flops
            layer_info['memory'] = memory
            total_flops += flops
            total_memory += memory
            self.model_info['layers'].append(layer_info)

        for var in concrete_func.graph.variables:
            var_info = {
                'name': var.name,
                'type': 'Variable',
                'shape': var.shape.as_list(),
                'dtype': var.dtype.name
            }
            self.model_info['layers'].append(var_info)
            self.model_info['total_params'] += reduce(operator.mul, var.shape.as_list(), 1)
            total_memory += reduce(operator.mul, var.shape.as_list(), 1) * 4  # Assuming float32

        self.model_info['compute_requirements'] = f"{total_flops / 1e9:.2f} GFLOPs"
        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def analyze_pytorch_model(self):
        if isinstance(self.model, dict):
            self.analyze_pytorch_state_dict()
        else:
            self.analyze_pytorch_full_model()

    def analyze_pytorch_state_dict(self):
        self.model_info['total_params'] = sum(p.numel() for p in self.model.values() if p.dim() > 0)
        total_memory = sum(p.numel() * p.element_size() for p in self.model.values())
        
        for name, param in self.model.items():
            layer_info = {
                'name': name,
                'type': 'Parameter',
                'shape': list(param.shape),
                'params': param.numel(),
                'memory': param.numel() * param.element_size()
            }
            self.model_info['layers'].append(layer_info)

        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def analyze_pytorch_state_dict(self):
        self.model_info['total_params'] = sum(p.numel() for p in self.model.values() if p.dim() > 1)
        total_memory = sum(p.numel() * p.element_size() for p in self.model.values())
        
        for name, param in self.model.items():
            if param.dim() > 1:  # Exclude 1D tensors which are usually biases
                layer_info = {
                    'name': name,
                    'type': 'Parameter',
                    'shape': list(param.shape),
                    'params': param.numel(),
                    'memory': param.numel() * param.element_size()
                }
                self.model_info['layers'].append(layer_info)

        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def analyze_pytorch_full_model(self):
        self.model_info['total_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Use a forward hook to capture the input/output shapes
        def hook_fn(module, input, output):
            if hasattr(module, 'weight'):
                flops, memory = self.estimate_layer_compute_memory(module, input, output)
            else:
                flops, memory = 0, 0
            layer_info = {
                'name': module.__class__.__name__,
                'type': module.__class__.__name__,
                'input_shape': tuple(input[0].shape) if isinstance(input, tuple) else input.shape,
                'output_shape': tuple(output.shape),
                'params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'flops': flops,
                'memory': memory
            }
            self.model_info['layers'].append(layer_info)

        hooks = []
        for layer in self.model.modules():
            if not isinstance(layer, torch.nn.Sequential) and len(list(layer.children())) == 0:
                hooks.append(layer.register_forward_hook(hook_fn))

        # Run a dummy forward pass
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # Adjust according to your model
            self.model(dummy_input)

        for hook in hooks:
            hook.remove()
        
        total_flops = sum(layer.get('flops', 0) for layer in self.model_info['layers'])
        total_memory = sum(layer.get('memory', 0) for layer in self.model_info['layers'])
        
        self.model_info['compute_requirements'] = f"{total_flops / 1e9:.2f} GFLOPs"
        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def analyze_onnx_model(self):
        self.model_info['total_params'] = sum(reduce(operator.mul, t.dims) for t in self.model.graph.initializer)
        total_flops = 0
        total_memory = 0
        
        for node in self.model.graph.node:
            layer_info = {
                'name': node.name,
                'type': node.op_type,
                'input': node.input,
                'output': node.output
            }
            flops, memory = self.estimate_onnx_node_compute_memory(node)
            layer_info['flops'] = flops
            layer_info['memory'] = memory
            total_flops += flops
            total_memory += memory
            self.model_info['layers'].append(layer_info)

        if self.model.graph.input:
            self.model_info['input_shape'] = [
                [d.dim_value if d.dim_value != 0 else None for d in input.type.tensor_type.shape.dim]
                for input in self.model.graph.input
            ]
        if self.model.graph.output:
            self.model_info['output_shape'] = [
                [d.dim_value if d.dim_value != 0 else None for d in output.type.tensor_type.shape.dim]
                for output in self.model.graph.output
            ]

        self.model_info['compute_requirements'] = f"{total_flops / 1e9:.2f} GFLOPs"
        self.model_info['memory_usage'] = f"{total_memory / 1e6:.2f} MB"
        self.estimate_bottlenecks()

    def estimate_layer_compute_memory(self, layer, input=None, output=None):
        # Handling Keras layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            k_h, k_w = layer.kernel_size
            input_shape = self.get_shape(layer.input)
            output_shape = self.get_shape(layer.output)
            if input_shape != 'Unknown' and len(input_shape) >= 4:
                in_c = input_shape[-1]
            else:
                in_c = layer.filters
            out_c = layer.filters
            if output_shape != 'Unknown' and len(output_shape) >= 4:
                out_h, out_w = output_shape[1:3]
            else:
                out_h = out_w = 1
            flops = k_h * k_w * in_c * out_c * out_h * out_w * 2
            memory = in_c * out_c * k_h * k_w * 4
        elif isinstance(layer, tf.keras.layers.Dense):
            input_shape = self.get_shape(layer.input)
            if input_shape != 'Unknown' and len(input_shape) >= 2:
                in_features = input_shape[-1]
            else:
                in_features = layer.units
            out_features = layer.units
            flops = in_features * out_features * 2
            memory = in_features * out_features * 4

        # Handling PyTorch layers
        elif isinstance(layer, torch.nn.Conv2d):
            k_h, k_w = layer.kernel_size
            in_c, out_c = layer.in_channels, layer.out_channels
            if input is not None and output is not None:
                out_h, out_w = output.shape[2], output.shape[3]
            else:
                out_h = out_w = 32  # Placeholder
            flops = k_h * k_w * in_c * out_c * out_h * out_w * 2
            memory = in_c * out_c * k_h * k_w * 4
        elif isinstance(layer, torch.nn.Linear):
            in_features, out_features = layer.in_features, layer.out_features
            flops = in_features * out_features * 2
            memory = in_features * out_features * 4
        else:
            return 0, 0
        return flops, memory

    def estimate_tf_node_compute_memory(self, node, graph):
        if node.op in ['Conv2D', 'MatMul']:
            input_shape = self.get_tensor_shape(graph, node.input[0])
            weight_shape = self.get_tensor_shape(graph, node.input[1])
            if input_shape and weight_shape:
                if node.op == 'Conv2D':
                    k_h, k_w = weight_shape[0], weight_shape[1]
                    in_c, out_c = weight_shape[2], weight_shape[3]
                    out_h, out_w = input_shape[1], input_shape[2]
                    flops = k_h * k_w * in_c * out_c * out_h * out_w * 2
                else:  # MatMul
                    flops = input_shape[-1] * weight_shape[-1] * input_shape[0] * 2
                memory = np.prod(weight_shape) * 4  # Assuming float32
                return flops, memory
        return 0, 0

    def estimate_onnx_node_compute_memory(self, node):
        if node.op_type in ['Conv', 'Gemm']:
            input_shape = self.get_onnx_input_shape(node.input[0])
            weight_shape = self.get_onnx_input_shape(node.input[1])
            if input_shape and weight_shape:
                if node.op_type == 'Conv':
                    k_h, k_w = weight_shape[2], weight_shape[3]
                    in_c, out_c = weight_shape[1], weight_shape[0]
                    out_h, out_w = input_shape[2], input_shape[3]
                    flops = k_h * k_w * in_c * out_c * out_h * out_w * 2
                else:  # Gemm
                    flops = input_shape[1] * weight_shape[0] * input_shape[0] * 2
                memory = np.prod(weight_shape) * 4  # Assuming float32
                return flops, memory
        return 0, 0

    def get_tensor_shape(self, graph, tensor_name):
        tensor = graph.get_tensor_by_name(tensor_name)
        if tensor.shape.dims:
            return [dim.size for dim in tensor.shape.dims]
        return None

    def get_onnx_input_shape(self, input_name):
        for input_info in self.model.graph.input:
            if input_info.name == input_name:
                return [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        for init in self.model.graph.initializer:
            if init.name == input_name:
                return list(init.dims)

    def estimate_bottlenecks(self):
        if not self.model_info['layers']:
            return

        compute_threshold = np.percentile([layer.get('flops', 0) for layer in self.model_info['layers']], 90)
        memory_threshold = np.percentile([layer.get('memory', 0) for layer in self.model_info['layers']], 90)

        for layer in self.model_info['layers']:
            if layer.get('flops', 0) > compute_threshold:
                self.model_info['bottlenecks'].append(f"High compute in {layer['name']}: {layer['flops']} FLOPs")
            if layer.get('memory', 0) > memory_threshold:
                self.model_info['bottlenecks'].append(f"High memory usage in {layer['name']}: {layer['memory']} bytes")
    
    def analyze_tensor_shapes(self) -> Dict[str, Any]:
            tensor_shapes = {}
            if self.model_format in ['tensorflow', 'keras']:
                for layer in self.model_info['layers']:
                    tensor_shapes[layer['name']] = {
                        'input_shape': layer.get('input_shape', 'Unknown'),
                        'output_shape': layer.get('output_shape', 'Unknown')
                    }
            elif self.model_format == 'pytorch':
                for layer in self.model_info['layers']:
                    tensor_shapes[layer['name']] = {
                        'shape': layer.get('shape', 'Unknown')
                    }
            elif self.model_format == 'onnx':
                for tensor in self.model.graph.initializer:
                    tensor_shapes[tensor.name] = list(tensor.dims)
            return tensor_shapes


    def identify_key_tensors(self) -> Dict[str, List[str]]:
        key_tensors = {'inputs': [], 'outputs': [], 'intermediate': []}
        if self.model_format in ['tensorflow', 'keras']:
            key_tensors['inputs'] = [layer['name'] for layer in self.model_info['layers'] if layer['type'] == 'InputLayer']
            key_tensors['outputs'] = [self.model_info['layers'][-1]['name']]
            key_tensors['intermediate'] = [layer['name'] for layer in self.model_info['layers'] if layer['name'] not in key_tensors['inputs'] + key_tensors['outputs']]
        elif self.model_format == 'pytorch':
            key_tensors['inputs'] = [self.model_info['layers'][0]['name']]
            key_tensors['outputs'] = [self.model_info['layers'][-1]['name']]
            key_tensors['intermediate'] = [layer['name'] for layer in self.model_info['layers'][1:-1]]
        elif self.model_format == 'onnx':
            key_tensors['inputs'] = [input.name for input in self.model.graph.input]
            key_tensors['outputs'] = [output.name for output in self.model.graph.output]
            key_tensors['intermediate'] = [node.name for node in self.model.graph.node]
        return key_tensors

    def analyze_data_transformations(self) -> List[Dict[str, Any]]:
        transformations = []
        if self.model_format in ['tensorflow', 'keras']:
            for layer in self.model_info['layers']:
                if layer['type'] in ['Reshape', 'Permute', 'Flatten']:
                    transformations.append({
                        'type': layer['type'],
                        'name': layer['name'],
                        'input_shape': layer.get('input_shape', 'Unknown'),
                        'output_shape': layer.get('output_shape', 'Unknown')
                    })
        elif self.model_format == 'pytorch':
            for layer in self.model_info['layers']:
                if layer.get('type') in ['Flatten', 'Unflatten', 'View']:
                    transformations.append({
                        'type': layer['type'],
                        'name': layer['name']
                    })
        elif self.model_format == 'onnx':
            for node in self.model.graph.node:
                if node.op_type in ['Reshape', 'Transpose', 'Flatten']:
                    transformations.append({
                        'type': node.op_type,
                        'name': node.name
                    })
        return transformations

    def run_analysis(self) -> Dict[str, Any]:
        self.load_model()
        self.analyze_model()
        tensor_shapes = self.analyze_tensor_shapes()
        key_tensors = self.identify_key_tensors()
        data_transformations = self.analyze_data_transformations()

        return {
            'model_info': self.model_info,
            'tensor_shapes': tensor_shapes,
            'key_tensors': key_tensors,
            'data_transformations': data_transformations
        }
