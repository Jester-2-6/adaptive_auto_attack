import io
from PIL import Image
import torch
import torchvision.transforms as transforms

def compress_tensor_batch(tensor_batch, quality=50):
    compressed_batch = []
    for tensor in tensor_batch:
        # Convert tensor to PIL Image (handles [C, H, W] format)
        img = transforms.ToPILImage()(tensor.cpu())
        
        # Compress to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        # Load compressed image
        img_compressed = Image.open(buffer)
        
        # Convert back to tensor
        tensor_compressed = transforms.ToTensor()(img_compressed)
        compressed_batch.append(tensor_compressed)
    
    # Stack tensors and move to original device
    compressed_batch = torch.stack(compressed_batch).to(tensor_batch.device)
    return compressed_batch
