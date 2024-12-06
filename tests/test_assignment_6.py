import torch
import sys
import os

# Add the parent directory to system path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assignment_6 import Net

def test_model_parameters():
    # Test total parameter count
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, "Model has too many parameters"

def test_batch_normalization():
    # Test presence of batch normalization
    model = Net()
    has_batchnorm = False
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            has_batchnorm = True
            break
    assert has_batchnorm, "Model should use batch normalization"

def test_dropout():
    # Test presence of dropout
    model = Net() 
    has_dropout = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            break
    assert has_dropout, "Model should use dropout"

def test_final_layer():
    # Test if model uses GAP or FC layer at the end
    model = Net()
    last_layers = list(model.modules())[-3:]  # Check last few layers
    has_gap = any(isinstance(layer, torch.nn.AdaptiveAvgPool2d) for layer in last_layers)
    assert has_gap, "Model should end with either Global Average Pooling or Fully Connected layer"
