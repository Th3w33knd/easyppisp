"""
04_optimization.py - Differentiable ISP optimization.

Demonstrates optimizing ISP parameters to minimize a custom loss, 
using the differentiable properties of EasyPPISP.
"""

import torch
import torch.nn.functional as F
from easyppisp.modules import ISPPipeline
from easyppisp.losses import exposure_mean_loss

def main():
    # 1. Setup a pipeline and a target (e.g., we want the image to have mean 0.5)
    pipeline = ISPPipeline()
    image = torch.rand(128, 128, 3)
    target_mean = 0.5
    
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.1)
    
    print(f"Initial mean: {pipeline(image).final.mean().item():.4f}")
    
    # 2. Optimization loop
    for i in range(50):
        optimizer.zero_grad()
        
        result = pipeline(image)
        
        # Style loss (simple mean match)
        loss_task = F.mse_loss(result.final.mean(), torch.tensor(target_mean))
        
        # Regularization loss to keep parameters plausible
        # (Example: keep exposure near 0 if possible)
        loss_reg = exposure_mean_loss(pipeline.exposure.delta_t)
        
        total_loss = loss_task + 0.1 * loss_reg
        total_loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f"Step {i+1:2d} | Loss: {total_loss.item():.6f} | Mean: {result.final.mean().item():.4f}")
            
    print("Optimization finished!")
    print(f"Final exposure: {pipeline.exposure.delta_t.item():.3f} EV")

if __name__ == "__main__":
    main()
