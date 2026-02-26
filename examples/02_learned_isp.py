"""
02_learned_isp.py - Using the ISPController for predictive ISP.

Demonstrates how to use the CNN-based controller to predict exposure
and color parameters for an input image.
"""

import torch
from easyppisp.modules import ISPController, ISPPipeline
from easyppisp.params import PipelineParams

def main():
    # 1. Initialize controller and a linear radiance image
    controller = ISPController()
    image = torch.rand(1, 256, 256, 3) # Batch of 1
    
    # 2. Predict parameters
    print("Predicting ISP parameters from global visual context...")
    preds = controller(image)
    
    exp_off = preds["exposure_offset"]      # (B, 1)
    color_flat = preds["color_params_flat"] # (B, 8)
    
    print(f"Predicted exposure: {exp_off.item():.3f} EV")
    
    # 3. Apply to image
    # Note: In a real training loop, you'd map color_flat to a dict
    # or use apply_pipeline functionally.
    pipeline = ISPPipeline()
    # (Simplified application for demonstration)
    result = pipeline(image[0]) 
    
    print(f"Processed image shape: {result.final.shape}")

if __name__ == "__main__":
    main()
