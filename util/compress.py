import torch
from torchvision import transforms
from PIL import Image
import io

DEFAULT_COMPRESSION_STRENGTH = 0.6


class CompressTensorBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, strength):
        compressed_images = []
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        if tensor.shape[0] == 0:
            return tensor

        for img in tensor:
            pil_img = to_pil(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=int(strength * 100))
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed_images.append(to_tensor(compressed_img))

        compressed_images = torch.stack(compressed_images).to(tensor.device)
        ctx.save_for_backward(tensor, compressed_images)
        ctx.strength = strength

        return compressed_images

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def compress_tensor_batch(
    tensor: torch.Tensor, strength: float = DEFAULT_COMPRESSION_STRENGTH
) -> torch.Tensor:
    return CompressTensorBatchFunction.apply(tensor, strength)


def compress_one_image(
    tensor: torch.Tensor, strength: float = DEFAULT_COMPRESSION_STRENGTH
) -> torch.Tensor:
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    pil_img = to_pil(tensor)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=int(strength * 100))
    buffer.seek(0)
    compressed_img = Image.open(buffer)

    return to_tensor(compressed_img)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import requests
    from PIL import Image

    response = requests.get("https://picsum.photos/32/32")
    img = Image.open(io.BytesIO(response.content))
    tensor = transforms.ToTensor()(img)

    # Display original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(np.transpose(tensor.numpy(), (1, 2, 0)))

    # Compress and display compressed image
    compressed_tensor = compress_one_image(tensor)
    plt.subplot(1, 2, 2)
    plt.title("Compressed Image")
    plt.imshow(np.transpose(compressed_tensor.numpy(), (1, 2, 0)))

    plt.show()

    # Save images to disk
    original_img = transforms.ToPILImage()(tensor)
    compressed_img = transforms.ToPILImage()(compressed_tensor)
    original_img.save("./original_image.jpg")
    compressed_img.save("./compressed_image.jpg")
