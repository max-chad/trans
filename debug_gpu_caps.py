import torch

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    count = torch.cuda.device_count()
    print(f"Found {count} CUDA devices")
    
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        major = props.major
        minor = props.minor
        print(f"Device {i}: {name}")
        print(f"  Compute Capability: {major}.{minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi Processor Count: {props.multi_processor_count}")

if __name__ == "__main__":
    check_gpu()
