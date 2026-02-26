"""
01_quickstart.py - Basic usage of EasyPPISP.

Demonstrates loading an image, applying simple exposure and presets,
and saving the results.
"""

import torch
import easyppisp
from easyppisp.utils import load_image, save_image

def main():
    # 1. Load a linear radiance image (HWC, float32)
    # If starting from sRGB, use easyppisp.utils.srgb_to_linear
    image = torch.rand(512, 512, 3) # Dummy linear image
    
    print("Applying +1.5 stops of exposure...")
    bright = easyppisp.apply(image, exposure=1.5)
    
    print("Applying 'kodak_portra_400' preset...")
    filmic = easyppisp.apply(image, preset="kodak_portra_400")
    
    # 2. Save results
    # save_image handles linear -> sRGB conversion and uint8 scaling
    save_image(bright, "output_bright.jpg")
    save_image(filmic, "output_filmic.jpg")
    print("Done! Check output_bright.jpg and output_filmic.jpg")

if __name__ == "__main__":
    main()
