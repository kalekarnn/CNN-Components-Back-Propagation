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
    assert total_params < 20000, "Model has too many parameters"

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
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_gap, "Model should use either GAP or Fully Connected layer"

def test_accuracy():
    """Test if model achieves accuracy above 99.4%"""
    try:
        with open('final_accuracy.txt', 'r') as f:
            accuracy = float(f.read().strip())
        assert accuracy >= 99.4, f"Model accuracy {accuracy}% should be greater than 99.4%"
    except FileNotFoundError:
        assert False, "final_accuracy.txt file not found. Make sure the notebook saves the final accuracy."
