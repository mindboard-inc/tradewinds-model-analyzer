import argparse
import logging
import traceback
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_model(model_path, output_file):
    logging.info(f"Analyzing model: {model_path}")
    try:
        file_extension = os.path.splitext(model_path)[1].lower()
        if file_extension == '.h5':
            results = analyze_h5_file(model_path)
        elif file_extension in ['.pt', '.pth', '.bin']:
            results = analyze_pytorch_file(model_path)
        else:
            raise ValueError("Unsupported file format. Please use .h5, .pt, .pth, or .bin files.")
        
        results = add_advanced_analytics(results)
        generate_visualizations(results, os.path.splitext(output_file)[0])
        dump_results_to_json(results, output_file)
        logging.info(f"Analysis results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error analyzing model: {str(e)}")
        logging.error(traceback.format_exc())

def analyze_h5_file(file_path):
    results = initialize_results(file_path, 'HDF5')

    with h5py.File(file_path, 'r') as f:
        results['file_info']['mode'] = f.mode
        results['file_info']['libver'] = f.libver

        def visit_item(name, obj):
            if isinstance(obj, h5py.Group):
                results['structure'].append({'name': name, 'type': 'Group'})
                layer_name = name.split('/')[-1]
                results['layer_types'][layer_name] += 1
            elif isinstance(obj, h5py.Dataset):
                process_dataset(results, name, obj)

        f.visititems(visit_item)

    finalize_results(results)
    return results

def analyze_pytorch_file(file_path):
    results = initialize_results(file_path, 'PyTorch')
    
    try:
        model = torch.load(file_path, map_location=torch.device('cpu'))
    except RuntimeError:
        # If torch.load fails, try loading as a state dict
        model = torch.load(file_path, map_location=torch.device('cpu'), pickle_module=torch)
    
    def visit_pytorch_dict(prefix, obj):
        for name, param in obj.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(param, dict):
                visit_pytorch_dict(full_name, param)
            elif isinstance(param, torch.Tensor):
                process_tensor(results, full_name, param)
            elif isinstance(param, np.ndarray):
                process_numpy_array(results, full_name, param)

    if isinstance(model, dict):
        visit_pytorch_dict('', model)
    elif hasattr(model, 'state_dict'):
        visit_pytorch_dict('', model.state_dict())
    else:
        raise ValueError("Unsupported PyTorch model format")

    finalize_results(results)
    return results

def initialize_results(file_path, model_type):
    return {
        'file_info': {
            'filename': file_path,
            'model_type': model_type
        },
        'structure': [],
        'layer_types': defaultdict(int),
        'total_parameters': 0,
        'estimated_memory_usage': 0,
        'data_types': defaultdict(int),
        'largest_layers': []
    }

def process_dataset(results, name, obj):
    dataset_info = {
        'name': name,
        'type': 'Dataset',
        'shape': list(obj.shape),
        'dtype': str(obj.dtype),
        'size': int(obj.size)
    }
    results['structure'].append(dataset_info)
    results['total_parameters'] += obj.size
    memory_usage = obj.size * obj.dtype.itemsize
    results['estimated_memory_usage'] += memory_usage
    results['data_types'][str(obj.dtype)] += 1
    results['largest_layers'].append((name, memory_usage))

def process_tensor(results, name, tensor):
    tensor_info = {
        'name': name,
        'type': 'Tensor',
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'size': int(tensor.numel())
    }
    results['structure'].append(tensor_info)
    results['total_parameters'] += tensor.numel()
    memory_usage = tensor.numel() * tensor.element_size()
    results['estimated_memory_usage'] += memory_usage
    results['data_types'][str(tensor.dtype)] += 1
    results['largest_layers'].append((name, memory_usage))

def process_numpy_array(results, name, array):
    array_info = {
        'name': name,
        'type': 'NumPy Array',
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'size': int(array.size)
    }
    results['structure'].append(array_info)
    results['total_parameters'] += array.size
    memory_usage = array.nbytes
    results['estimated_memory_usage'] += memory_usage
    results['data_types'][str(array.dtype)] += 1
    results['largest_layers'].append((name, memory_usage))

def finalize_results(results):
    results['estimated_memory_usage'] = float(results['estimated_memory_usage'] / (1024 * 1024))
    results['largest_layers'] = sorted(results['largest_layers'], key=lambda x: x[1], reverse=True)[:5]
    results['largest_layers'] = [{'name': name, 'size_mb': float(size / (1024 * 1024))} for name, size in results['largest_layers']]
    results['layer_types'] = dict(results['layer_types'])
    results['data_types'] = dict(results['data_types'])

def dump_results_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def add_advanced_analytics(results):
    # Analyze parameter distribution
    param_sizes = [layer['size'] for layer in results['structure'] if layer['type'] in ['Dataset', 'Tensor']]
    results['parameter_statistics'] = {
        'mean': float(np.mean(param_sizes)),
        'median': float(np.median(param_sizes)),
        'std': float(np.std(param_sizes)),
        'min': int(np.min(param_sizes)),
        'max': int(np.max(param_sizes))
    }
    
    # Identify potential bottlenecks
    memory_threshold = float(np.percentile([layer['size_mb'] for layer in results['largest_layers']], 90))
    results['potential_bottlenecks'] = [
        layer for layer in results['largest_layers'] 
        if layer['size_mb'] > memory_threshold
    ]
    
    # Analyze model complexity
    results['model_complexity'] = {
        'total_parameters': int(results['total_parameters']),
        'parameter_to_memory_ratio': float(results['total_parameters'] / results['estimated_memory_usage']),
        'average_tensor_size': float(np.mean([layer['size'] for layer in results['structure'] if layer['type'] in ['Dataset', 'Tensor']]))
    }
    
    # Identify unusual layer patterns
    layer_sizes = [layer['size'] for layer in results['structure'] if layer['type'] in ['Dataset', 'Tensor']]
    z_scores = stats.zscore(layer_sizes)
    results['unusual_layers'] = [
        results['structure'][i]['name'] for i, z in enumerate(z_scores) if abs(z) > 2
    ]
    
    return results

def generate_visualizations(results, output_prefix):
    # Parameter size distribution
    plt.figure(figsize=(10, 6))
    sns.histplot([layer['size'] for layer in results['structure'] if layer['type'] in ['Dataset', 'Tensor']], kde=True)
    plt.title('Distribution of Parameter Sizes')
    plt.xlabel('Parameter Size')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_prefix}_param_distribution.png")
    plt.close()

    # Layer sizes
    plt.figure(figsize=(12, 6))
    layer_names = [layer['name'].split('.')[-1] for layer in results['largest_layers']]
    layer_sizes = [layer['size_mb'] for layer in results['largest_layers']]
    sns.barplot(x=layer_sizes, y=layer_names)
    plt.title('Top 5 Largest Layers')
    plt.xlabel('Size (MB)')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_largest_layers.png")
    plt.close()

    # Data type distribution
    plt.figure(figsize=(10, 6))
    data_types = list(results['data_types'].keys())
    data_type_counts = list(results['data_types'].values())
    plt.pie(data_type_counts, labels=data_types, autopct='%1.1f%%')
    plt.title('Distribution of Data Types')
    plt.savefig(f"{output_prefix}_datatype_distribution.png")
    plt.close()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def dump_results_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a TensorFlow (.h5) or PyTorch (.pt, .pth, .bin) model file and output results to JSON.")
    parser.add_argument("model_path", help="Path to the model file to analyze")
    parser.add_argument("--output", default="analysis_results.json", help="Output JSON file path")
    args = parser.parse_args()

    try:
        analyze_model(args.model_path, args.output)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
