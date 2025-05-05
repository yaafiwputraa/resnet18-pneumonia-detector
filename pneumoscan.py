"""
PneumoScan: X-Ray Classification for Pneumonia Detection
=======================================================
This project fine-tunes a pre-trained ResNet-18 model to classify chest X-ray images
into two categories: normal lungs and those affected by pneumonia.

Author: [Your Name]
Date: May 2025
"""

# Import required libraries
# --------------------------------------------------------------
# Data loading and manipulation
import os
import random
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# For progress bar
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------

class ChestXRayDataset(Dataset):
    """
    Custom Dataset class for the Chest X-Ray dataset.
    Adds functionality to view original image, visualize transformations, etc.
    """
    def __init__(self, root_dir, transform=None, phase='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            phase (string): 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.phase = phase
        self.classes = ['NORMAL', 'PNEUMONIA']
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for i, label in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, label)
            class_files = os.listdir(class_dir)
            class_paths = [os.path.join(class_dir, f) for f in class_files]
            self.image_paths.extend(class_paths)
            self.labels.extend([i] * len(class_paths))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_original_image(self, idx):
        """
        Get the original image without transformations
        """
        img_path = self.image_paths[idx]
        return Image.open(img_path).convert('RGB')
    
    def get_class_counts(self):
        """
        Return the count of images for each class
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip([self.classes[i] for i in unique], counts))


def prepare_data(data_zip='data/chestxrays.zip', use_validation=True, val_split=0.2):
    """
    Extract data and prepare DataLoaders
    """
    # Extract the dataset if needed
    if not os.path.exists('data/chestxrays'):
        print(f"Extracting dataset from {data_zip}...")
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall('data')
        print("Extraction completed.")
    else:
        print("Dataset already extracted.")
    
    # Define transformations
    # For training: includes data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # For validation/testing: only resize and normalize
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_full_dataset = ChestXRayDataset('data/chestxrays', transform=train_transform, phase='train')
    test_dataset = ChestXRayDataset('data/chestxrays', transform=val_test_transform, phase='test')
    
    # Create validation set if needed
    if use_validation:
        # Create validation dataset from training dataset
        val_size = int(len(train_full_dataset) * val_split)
        train_size = len(train_full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        
        # Update the transform for the validation dataset
        val_dataset.dataset.transform = val_test_transform
        
        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images")
    else:
        train_dataset = train_full_dataset
        val_dataset = None
        print(f"Training set: {len(train_dataset)} images")
    
    print(f"Test set: {len(test_dataset)} images")
    
    # Check class balance
    print("\nClass distribution in training set:")
    class_counts = train_full_dataset.get_class_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if use_validation:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, train_full_dataset, test_dataset


def visualize_dataset_samples(dataset, num_samples=5, classes=None):
    """
    Visualize random samples from the dataset
    """
    if classes is None:
        classes = dataset.classes
    
    # Create figure
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 6))
    
    # For each class
    for i, class_name in enumerate(classes):
        # Get indices for this class
        indices = [j for j, label in enumerate(dataset.labels) if label == i]
        
        # Randomly sample from these indices
        sample_indices = random.sample(indices, min(num_samples, len(indices)))
        
        # Plot each sample
        for j, idx in enumerate(sample_indices):
            img = dataset.get_original_image(idx)
            axes[i, j].imshow(img)
            axes[i, j].set_title(class_name)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------------------------------------------
# Model Building
# --------------------------------------------------------------

def create_model(model_type='resnet18', num_classes=2, feature_extract=True):
    """
    Create a pre-trained model for fine-tuning
    
    Args:
        model_type (str): Type of model to use ('resnet18', 'resnet34', 'densenet121')
        num_classes (int): Number of classes to classify
        feature_extract (bool): If True, only update the reshaped layer params
    
    Returns:
        model: The specified model with modified head
    """
    model = None
    
    if model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Move model to the device
    model = model.to(device)
    
    return model


def train_model(model, train_loader, criterion, optimizer, scheduler=None, 
                num_epochs=3, val_loader=None, early_stopping_patience=5):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        val_loader: DataLoader for validation data
        early_stopping_patience: Number of epochs to wait for improvement before stopping
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Track best model and early stopping variables
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    # Initialize history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            elif phase == 'val' and val_loader is not None:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            else:
                continue
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f'{phase} iter'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:  # phase == 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Update scheduler if needed
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                
                # Check if this is the best model so far
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
        
        # Check for early stopping
        if val_loader is not None and no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Load best model weights
    if val_loader is not None:
        print(f'Best val Acc: {best_val_acc:.4f}')
        model.load_state_dict(best_model_wts)
    
    return model, history


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy
    """
    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------------------------------------------
# Model Evaluation
# --------------------------------------------------------------

def evaluate_model(model, dataloader, criterion=None):
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function (optional)
    
    Returns:
        metrics: Dictionary containing evaluation metrics
        all_preds: All predictions
        all_labels: All true labels
        all_probs: All output probabilities
    """
    # Set model to evaluate mode
    model.eval()
    
    # Initialize variables
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    if criterion is not None:
        metrics['loss'] = running_loss / len(dataloader.dataset)
    
    # Calculate accuracy
    metrics['accuracy'] = np.mean(all_preds == all_labels)
    
    # Get classification report
    metrics['classification_report'] = classification_report(
        all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA'], output_dict=True
    )
    
    # Calculate ROC and AUC (for binary classification)
    if len(np.unique(all_labels)) == 2:
        # Use probability of positive class (pneumonia)
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        
        # Calculate precision-recall
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['average_precision'] = average_precision_score(all_labels, all_probs[:, 1])
    
    return metrics, all_preds, all_labels, all_probs


def plot_confusion_matrix(all_labels, all_preds, classes=None):
    """
    Plot confusion matrix
    """
    if classes is None:
        classes = ['NORMAL', 'PNEUMONIA']
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(precision, recall, average_precision):
    """
    Plot precision-recall curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, dataset, num_images=8, classes=None):
    """
    Visualize some predictions
    """
    if classes is None:
        classes = dataset.classes
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_images)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Set model to evaluation mode
    model.eval()
    
    for i, idx in enumerate(indices):
        # Get image and label
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(img)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
        
        # Get original image for display
        orig_img = dataset.get_original_image(idx)
        
        # Create subplot
        ax = plt.subplot(int(num_images/4), 4, i + 1)
        
        # Display image
        ax.imshow(orig_img)
        
        # Set title with true label, prediction, and confidence
        title = f"True: {classes[label]}\nPred: {classes[pred.item()]}"
        confidence = probs[pred.item()].item() * 100
        title += f"\nConf: {confidence:.1f}%"
        
        # Color title based on correctness
        color = "green" if pred.item() == label else "red"
        ax.set_title(title, color=color)
        
        # Turn off axis
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def gradcam_visualization(model, dataset, num_images=6, classes=None):
    """
    Visualize Grad-CAM activation maps for the model
    
    Note: This requires the pytorch-grad-cam package
    pip install pytorch-grad-cam
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam not installed. Run: pip install pytorch-grad-cam")
        return
    
    if classes is None:
        classes = dataset.classes
    
    # Get random indices, half normal and half pneumonia
    normal_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    pneumonia_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
    
    normal_samples = random.sample(normal_indices, num_images // 2)
    pneumonia_samples = random.sample(pneumonia_indices, num_images // 2)
    indices = normal_samples + pneumonia_samples
    
    # Define the target layer for GradCAM (usually the last convolutional layer)
    if isinstance(model, models.resnet.ResNet):
        target_layer = model.layer4[-1]
    else:
        print("Unsupported model type for GradCAM")
        return
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        # Get image and label
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get original image for display
        orig_img = np.array(dataset.get_original_image(idx)) / 255.0  # Normalize to [0,1]
        
        # Define target (the predicted class)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
        
        targets = [ClassifierOutputTarget(pred.item())]
        
        # Generate GradCAM
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # First image in batch
        
        # Overlay GradCAM on original image
        cam_image = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
        
        # Create subplots
        # Original image
        ax1 = plt.subplot(num_images, 2, i*2 + 1)
        ax1.imshow(orig_img)
        ax1.set_title(f"Original: {classes[label]}")
        ax1.axis('off')
        
        # GradCAM visualization
        ax2 = plt.subplot(num_images, 2, i*2 + 2)
        ax2.imshow(cam_image)
        ax2.set_title(f"GradCAM: {classes[pred.item()]}")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------------------

def hyperparameter_search(train_loader, val_loader, num_epochs=3, early_stopping_patience=5):
    """
    Perform a small hyperparameter search
    
    Returns:
        results_df: DataFrame with hyperparameter search results
    """
    # Define hyperparameters to search
    learning_rates = [0.01, 0.001, 0.0001]
    optimizers = ['adam', 'sgd']
    model_types = ['resnet18', 'resnet34']
    
    # Initialize results list
    results = []
    
    # Iterate through hyperparameter combinations
    for model_type in model_types:
        for optimizer_name in optimizers:
            for lr in learning_rates:
                print(f"\nTrying: model={model_type}, optimizer={optimizer_name}, lr={lr}")
                
                # Create model
                model = create_model(model_type=model_type)
                
                # Define loss function
                criterion = nn.CrossEntropyLoss()
                
                # Define optimizer
                if optimizer_name.lower() == 'adam':
                    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
                elif optimizer_name.lower() == 'sgd':
                    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
                
                # Define scheduler
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
                
                # Train model
                _, history = train_model(
                    model, train_loader, criterion, optimizer, scheduler,
                    num_epochs=num_epochs, val_loader=val_loader, 
                    early_stopping_patience=early_stopping_patience
                )
                
                # Evaluate model
                metrics, _, _, _ = evaluate_model(model, val_loader, criterion)
                
                # Record results
                results.append({
                    'model_type': model_type,
                    'optimizer': optimizer_name,
                    'learning_rate': lr,
                    'val_accuracy': metrics['accuracy'],
                    'val_f1': metrics['classification_report']['weighted avg']['f1-score'],
                    'val_loss': metrics['loss']
                })
                
                # Clear memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print sorted results
    print("\nHyperparameter search results (sorted by validation accuracy):")
    print(results_df.sort_values('val_accuracy', ascending=False))
    
    # Save results
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    
    return results_df


# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------

def main():
    """
    Main function to execute the pipeline
    """
    print("PneumoScan: X-Ray Classification for Pneumonia Detection")
    print("=" * 60)
    
    # Setup data
    train_loader, val_loader, test_loader, train_dataset, test_dataset = prepare_data(
        use_validation=True, val_split=0.2
    )
    
    # Visualize data samples
    print("\nVisualizing dataset samples...")
    visualize_dataset_samples(train_dataset, num_samples=5)
    
    # Create model
    print("\nCreating model...")
    model = create_model(model_type='resnet18', num_classes=2, feature_extract=True)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Train model
    print("\nTraining model...")
    trained_model, history = train_model(
        model, train_loader, criterion, optimizer, scheduler,
        num_epochs=10, val_loader=val_loader, early_stopping_patience=5
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_metrics, test_preds, test_labels, test_probs = evaluate_model(trained_model, test_loader, criterion)
    
    # Print evaluation results
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test F1 Score: {test_metrics['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds)
    
    # Plot ROC curve
    if 'roc_auc' in test_metrics:
        print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
        plot_roc_curve(test_metrics['fpr'], test_metrics['tpr'], test_metrics['roc_auc'])
    
    # Plot precision-recall curve
    if 'average_precision' in test_metrics:
        print(f"Average Precision: {test_metrics['average_precision']:.4f}")
        plot_precision_recall_curve(
            test_metrics['precision'], test_metrics['recall'], test_metrics['average_precision']
        )
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(trained_model, test_dataset, num_images=8)
    
    # Visualize GradCAM (if available)
    try:
        print("\nGenerating GradCAM visualizations...")
        gradcam_visualization(trained_model, test_dataset, num_images=6)
    except Exception as e:
        print(f"Error generating GradCAM: {e}")
    
    # Save model
    print("\nSaving model...")
    torch.save(trained_model.state_dict(), 'pneumoscan_model.pth')
    
    # Optional: Run hyperparameter search
    run_hyperparam_search = False  # Set to True to run hyperparameter search
    if run_hyperparam_search:
        print("\nRunning hyperparameter search...")
        results_df = hyperparameter_search(train_loader, val_loader, num_epochs=3)
        print("\nBest hyperparameters:")
        best_row = results_df.loc[results_df['val_accuracy'].idxmax()]
        print(best_row)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()