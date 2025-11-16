import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    # Test 1: Weighted Cross-Entropy Loss
    print("\n" + "="*80)
    print("TEST 1: Weighted Cross-Entropy Loss")
    print("="*80)
    
    class_weights = torch.FloatTensor([1.5, 1.0, 2.0])  # Example weights
    criterion = WeightedCrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, targets)
    
    print(f"Class weights: {class_weights}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test 2: Focal Loss
    print("\n" + "="*80)
    print("TEST 2: Focal Loss")
    print("="*80)
    
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    focal_loss = focal_criterion(logits, targets)
    
    print(f"Focal Loss (gamma=2.0): {focal_loss.item():.4f}")
    
    # Test 3: Accuracy
    print("\n" + "="*80)
    print("TEST 3: Accuracy")
    print("="*80)
    
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc.item()*100:.2f}%")
    
    # Test 4: Per-class metrics
    print("\n" + "="*80)
    print("TEST 4: Per-Class Metrics")
    print("="*80)
    
    metrics = compute_per_class_metrics(predictions, targets, num_classes=3)
    
    class_names = ['Alert', 'Low Vigilance', 'Drowsy']
    for cls in range(num_classes):
        print(f"\n{class_names[cls]} (class {cls}):")
        print(f"  Precision: {metrics['precision'][cls]:.3f}")
        print(f"  Recall:    {metrics['recall'][cls]:.3f}")
        print(f"  F1 Score:  {metrics['f1'][cls]:.3f}")
    
    # Test 5: Confusion Matrix
    print("\n" + "="*80)
    print("TEST 5: Confusion Matrix")
    print("="*80)
    
    cm = compute_confusion_matrix(predictions, targets, num_classes=3)
    print("\nConfusion Matrix:")
    print("         Predicted")
    print("       ", " ".join([f"{i:4d}" for i in range(num_classes)]))
    print("      ", "-" * (num_classes * 5))
    for i in range(num_classes):
        print(f"True {i} |", " ".join([f"{cm[i, j]:4d}" for j in range(num_classes)]))
    
    print("\n" + "="*80)
    print("✅ All tests completed successfully!")
    print("="*80)