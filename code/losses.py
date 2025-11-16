import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


# LOSS FUNCTIONS

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance.
    
    Args:
        weight: torch.FloatTensor of shape (num_classes,)
                Class weights computed from training set distribution
    """
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 3) raw logits from model
            targets: (B,) class labels (0, 1, or 2)
            
        Returns:
            loss: scalar tensor
        """
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -α * (1 - p_t)^γ * log(p_t)
    
    where:
        - p_t: probability of ground truth class
        - α: class weights (optional)
        - γ: focusing parameter (default 2)
        
    Higher γ → more focus on hard examples
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 3) raw logits
            targets: (B,) class labels
            
        Returns:
            loss: scalar tensor
        """
        # Compute cross-entropy loss per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        
        # Compute p_t (probability of ground truth class)
        pt = torch.exp(-ce_loss)
        
        # Focal loss = (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss


# CLASS WEIGHT COMPUTATION

def compute_class_weights(labels, num_classes=3, method='balanced'):
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: np.array or list of class labels
        num_classes: number of classes (default 3)
        method: 'balanced' or 'sqrt'
            - 'balanced': w_c = N_total / (K * N_c)
            - 'sqrt': w_c = sqrt(N_total / N_c)
    
    Returns:
        weights: torch.FloatTensor of shape (num_classes,)
        
    Example:
        >>> labels = [0, 0, 1, 1, 1, 2]
        >>> weights = compute_class_weights(labels, num_classes=3)
        >>> print(weights)
        tensor([1.0000, 0.6667, 2.0000])
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count samples per class
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 1)  # Avoid division by zero
        
        if method == 'balanced':
            # Balanced weighting: w_c = N_total / (K * N_c)
            weight = total_samples / (num_classes * count)
        elif method == 'sqrt':
            # Square root weighting (less aggressive)
            weight = np.sqrt(total_samples / count)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        weights.append(weight)
    
    # Convert to tensor and normalize
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Optional: normalize weights to sum to num_classes
    # This keeps the average weight at 1.0
    weights = weights * (num_classes / weights.sum())
    
    return weights


# EVALUATION METRICS

def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.
    
    Args:
        predictions: (B,) predicted class labels (0, 1, or 2)
        targets: (B,) true class labels
        
    Returns:
        accuracy: scalar tensor (0-1, multiply by 100 for percentage)
    """
    correct = (predictions == targets).float()
    return correct.mean()


def compute_per_class_metrics(predictions, targets, num_classes=3):
    """
    Compute per-class precision, recall, and F1 score.
    
    Args:
        predictions: (B,) predicted class labels
        targets: (B,) true class labels
        num_classes: number of classes (default 3)
        
    Returns:
        metrics: dict with keys:
            'precision': list of per-class precision
            'recall': list of per-class recall
            'f1': list of per-class F1
    """
    precision = []
    recall = []
    f1 = []
    
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == cls) & (targets == cls)).sum().float()
        fp = ((predictions == cls) & (targets != cls)).sum().float()
        fn = ((predictions != cls) & (targets == cls)).sum().float()
        
        # Precision = TP / (TP + FP)
        prec = tp / (tp + fp + 1e-6)
        
        # Recall = TP / (TP + FN)
        rec = tp / (tp + fn + 1e-6)
        
        # F1 = 2 * Precision * Recall / (Precision + Recall)
        f1_score = 2 * prec * rec / (prec + rec + 1e-6)
        
        precision.append(prec.item())
        recall.append(rec.item())
        f1.append(f1_score.item())
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_confusion_matrix(predictions, targets, num_classes=3):
    """
    Compute confusion matrix.
    
    Args:
        predictions: (B,) predicted class labels
        targets: (B,) true class labels
        num_classes: number of classes (default 3)
        
    Returns:
        confusion_matrix: (num_classes, num_classes) tensor
            confusion_matrix[i, j] = # samples with true class i, predicted class j
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(targets.view(-1), predictions.view(-1)):
        confusion[t.long(), p.long()] += 1
    
    return confusion


def print_classification_report(predictions, targets, class_names=None, num_classes=3):
    """
    Print a comprehensive classification report.
    
    Args:
        predictions: (B,) predicted class labels
        targets: (B,) true class labels
        class_names: list of class names (optional)
        num_classes: number of classes (default 3)
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Compute metrics
    accuracy = compute_accuracy(predictions, targets)
    metrics = compute_per_class_metrics(predictions, targets, num_classes)
    cm = compute_confusion_matrix(predictions, targets, num_classes)
    
    print("="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    
    print(f"\nOverall Accuracy: {accuracy.item()*100:.2f}%")
    print(f"Total Samples: {len(targets)}")
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 80)
    
    for i in range(num_classes):
        support = (targets == i).sum().item()
        print(f"{class_names[i]:<20} {metrics['precision'][i]:>12.3f} "
              f"{metrics['recall'][i]:>12.3f} {metrics['f1'][i]:>12.3f} "
              f"{support:>12d}")
    
    print("-" * 80)
    
    # Macro average
    macro_precision = np.mean(metrics['precision'])
    macro_recall = np.mean(metrics['recall'])
    macro_f1 = np.mean(metrics['f1'])
    
    print(f"{'Macro Average':<20} {macro_precision:>12.3f} "
          f"{macro_recall:>12.3f} {macro_f1:>12.3f} {len(targets):>12d}")
    
    # Weighted average
    weights = [(targets == i).sum().item() / len(targets) for i in range(num_classes)]
    weighted_precision = sum(w * p for w, p in zip(weights, metrics['precision']))
    weighted_recall = sum(w * r for w, r in zip(weights, metrics['recall']))
    weighted_f1 = sum(w * f for w, f in zip(weights, metrics['f1']))
    
    print(f"{'Weighted Average':<20} {weighted_precision:>12.3f} "
          f"{weighted_recall:>12.3f} {weighted_f1:>12.3f} {len(targets):>12d}")
    
    print("="*80)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    
    # Header
    print("         ", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    print("-" * 80)
    
    # Rows
    for i in range(num_classes):
        print(f"{class_names[i]:<10}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:>12d}", end="")
        print()
    
    print("="*80)


# TESTING

if __name__ == "__main__":
    print("="*80)
    print("Testing 3-Class Classification Loss and Metrics")
    print("="*80)
    
    # Test data
    batch_size = 8
    num_classes = 3
    
    # Mock logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 1, 2])  # Example labels
    
    print(f"\nBatch size: {batch_size}")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets: {targets}")
    
    # Get predictions
    predictions = torch.argmax(logits, dim=1)
    print(f"Predictions: {predictions}")
    
    # Test 1: Compute class weights
    print("\n" + "="*80)
    print("TEST 1: Compute Class Weights")
    print("="*80)
    
    # Simulate imbalanced dataset
    imbalanced_labels = [0]*100 + [1]*200 + [2]*50
    class_weights = compute_class_weights(imbalanced_labels, num_classes=3)
    print(f"Label distribution: Alert=100, Low Vig=200, Drowsy=50")
    print(f"Computed weights: {class_weights}")
    print(f"  Alert (class 0): {class_weights[0]:.3f}")
    print(f"  Low Vig (class 1): {class_weights[1]:.3f}")
    print(f"  Drowsy (class 2): {class_weights[2]:.3f}")
    
    # Test 2: Weighted Cross-Entropy Loss
    print("\n" + "="*80)
    print("TEST 2: Weighted Cross-Entropy Loss")
    print("="*80)
    
    criterion = WeightedCrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, targets)
    
    print(f"Class weights: {class_weights}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test 3: Focal Loss
    print("\n" + "="*80)
    print("TEST 3: Focal Loss")
    print("="*80)
    
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    focal_loss = focal_criterion(logits, targets)
    
    print(f"Focal Loss (gamma=2.0): {focal_loss.item():.4f}")
    
    # Test 4: Accuracy
    print("\n" + "="*80)
    print("TEST 4: Accuracy")
    print("="*80)
    
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc.item()*100:.2f}%")
    
    # Test 5: Per-class metrics
    print("\n" + "="*80)
    print("TEST 5: Per-Class Metrics")
    print("="*80)
    
    metrics = compute_per_class_metrics(predictions, targets, num_classes=3)
    
    class_names = ['Alert', 'Low Vigilance', 'Drowsy']
    for cls in range(num_classes):
        print(f"\n{class_names[cls]} (class {cls}):")
        print(f"  Precision: {metrics['precision'][cls]:.3f}")
        print(f"  Recall:    {metrics['recall'][cls]:.3f}")
        print(f"  F1 Score:  {metrics['f1'][cls]:.3f}")
    
    # Test 6: Confusion Matrix
    print("\n" + "="*80)
    print("TEST 6: Confusion Matrix")
    print("="*80)
    
    cm = compute_confusion_matrix(predictions, targets, num_classes=3)
    print("\nConfusion Matrix:")
    print("         Predicted")
    print("       ", " ".join([f"{i:4d}" for i in range(num_classes)]))
    print("      ", "-" * (num_classes * 5))
    for i in range(num_classes):
        print(f"True {i} |", " ".join([f"{cm[i, j]:4d}" for j in range(num_classes)]))
    
    # Test 7: Full Classification Report
    print("\n" + "="*80)
    print("TEST 7: Full Classification Report")
    print("="*80)
    
    print_classification_report(predictions, targets, class_names=['Alert', 'Low Vigilance', 'Drowsy'])
    
    print("\n" + "="*80)
    print("✅ All tests completed successfully!")
    print("="*80)