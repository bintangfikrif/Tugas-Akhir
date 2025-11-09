"""
Ordinal Regression Loss Functions for KSS Prediction

References:
- Coral Loss: Cao et al. "Rank consistent ordinal regression for neural networks 
  with application to age estimation" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss using binary cross-entropy.
    
    For K classes (0, 1, ..., K-1), we train K-1 binary classifiers:
    - Classifier i predicts P(Y > i)
    
    For 9-class KSS (0-8), we have 8 binary classifiers.
    
    Args:
        num_classes: Number of ordinal classes (default: 9 for KSS)
    """
    
    def __init__(self, num_classes=9):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # K-1 binary classifiers
        
    def forward(self, logits, ordinal_labels):
        """
        Args:
            logits: (B, K-1) raw logits from model (8 values for 9 classes)
            ordinal_labels: (B, K-1) binary labels (0 or 1)
            
        Returns:
            loss: scalar tensor
        """
        # Binary cross-entropy with logits
        # logits: (B, 8), ordinal_labels: (B, 8)
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            ordinal_labels, 
            reduction='mean'
        )
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Ordinal Regression + Cross-Entropy
    
    Useful for better calibration and to ensure the model learns
    both the ordinal structure and the actual class probabilities.
    """
    
    def __init__(self, num_classes=9, ordinal_weight=1.0, ce_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.ordinal_loss = OrdinalRegressionLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ordinal_weight = ordinal_weight
        self.ce_weight = ce_weight
        
    def forward(self, ordinal_logits, class_logits, ordinal_labels, class_labels):
        """
        Args:
            ordinal_logits: (B, K-1) logits for ordinal regression
            class_logits: (B, K) logits for classification
            ordinal_labels: (B, K-1) binary ordinal labels
            class_labels: (B,) class labels
            
        Returns:
            total_loss: weighted combination
            loss_dict: dictionary with individual losses
        """
        ord_loss = self.ordinal_loss(ordinal_logits, ordinal_labels)
        ce_loss = self.ce_loss(class_logits, class_labels)
        
        total_loss = self.ordinal_weight * ord_loss + self.ce_weight * ce_loss
        
        return total_loss, {
            'ordinal_loss': ord_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item()
        }


def logits_to_class(ordinal_logits):
    """
    Convert ordinal logits to predicted class.
    
    The predicted class is the number of binary classifiers that predict 1.
    
    Args:
        ordinal_logits: (B, K-1) raw logits
        
    Returns:
        predicted_classes: (B,) predicted class labels (0 to K-1)
    """
    # Apply sigmoid to get probabilities
    probas = torch.sigmoid(ordinal_logits)  # (B, K-1)
    
    # Count how many classifiers predict P(Y > i) > 0.5
    predictions = (probas > 0.5).long()  # (B, K-1)
    predicted_classes = predictions.sum(dim=1)  # (B,)
    
    return predicted_classes


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error for ordinal regression.
    
    Args:
        predictions: (B,) predicted class labels
        targets: (B,) true class labels
        
    Returns:
        mae: scalar tensor
    """
    return torch.abs(predictions - targets).float().mean()


def compute_accuracy_with_tolerance(predictions, targets, tolerance=0):
    """
    Compute accuracy with tolerance.
    
    Args:
        predictions: (B,) predicted class labels
        targets: (B,) true class labels
        tolerance: int,允許的誤差範圍 (0 means exact match)
        
    Returns:
        accuracy: scalar tensor
    """
    diff = torch.abs(predictions - targets)
    correct = (diff <= tolerance).float()
    return correct.mean()


if __name__ == "__main__":
    # Test ordinal regression loss
    print("="*60)
    print("Testing Ordinal Regression Loss")
    print("="*60)
    
    batch_size = 4
    num_classes = 9
    num_thresholds = num_classes - 1  # 8
    
    # Simulate model outputs
    ordinal_logits = torch.randn(batch_size, num_thresholds)  # (4, 8)
    
    # Simulate true labels (e.g., KSS levels 3, 5, 7, 2 -> classes 2, 4, 6, 1)
    true_classes = torch.tensor([2, 4, 6, 1])  # (4,)
    
    # Create ordinal labels
    ordinal_labels = torch.zeros(batch_size, num_thresholds)
    for i, cls in enumerate(true_classes):
        for j in range(cls):
            ordinal_labels[i, j] = 1.0
    
    print(f"True classes: {true_classes.tolist()}")
    print(f"Ordinal labels shape: {ordinal_labels.shape}")
    print(f"Ordinal labels:\n{ordinal_labels}")
    
    # Compute loss
    criterion = OrdinalRegressionLoss(num_classes=9)
    loss = criterion(ordinal_logits, ordinal_labels)
    print(f"\nOrdinal Regression Loss: {loss.item():.4f}")
    
    # Get predictions
    predicted_classes = logits_to_class(ordinal_logits)
    print(f"\nPredicted classes: {predicted_classes.tolist()}")
    
    # Compute metrics
    mae = compute_mae(predicted_classes, true_classes)
    acc = compute_accuracy_with_tolerance(predicted_classes, true_classes, tolerance=0)
    acc_tol1 = compute_accuracy_with_tolerance(predicted_classes, true_classes, tolerance=1)
    
    print(f"\nMetrics:")
    print(f"  MAE: {mae.item():.4f}")
    print(f"  Accuracy (exact): {acc.item()*100:.2f}%")
    print(f"  Accuracy (±1): {acc_tol1.item()*100:.2f}%")
    
    print("\n" + "="*60)
    print("Testing Combined Loss")
    print("="*60)
    
    # Test combined loss
    class_logits = torch.randn(batch_size, num_classes)  # (4, 9)
    combined_criterion = CombinedLoss(num_classes=9, ordinal_weight=1.0, ce_weight=0.1)
    
    total_loss, loss_dict = combined_criterion(
        ordinal_logits, class_logits, ordinal_labels, true_classes
    )
    
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
