import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import time

# Imports for Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Configuration & Data Loading ---
# Check if a GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for the images
# SqueezeNet expects 227x227 images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'processed_images'
train_dir = os.path.join(data_dir, 'train')

# This block now correctly checks for the train/val split BEFORE trying to load from it.
if os.path.exists(train_dir):
    # This part runs if you have already created train/val subfolders
    print("Found existing train/val split.")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
else:
    # This part runs for you: it finds no train/val folders and creates the split automatically.
    print("Train/val split not found. Creating it automatically from the main folder...")
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

    # Create the 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # IMPORTANT: Apply the correct 'validation' transforms to the validation dataset
    val_dataset.dataset.transform = data_transforms['val']

    # Create the data loaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes

print(f"Class names: {class_names}")
# --- 2. Handle Class Imbalance ---
print("Calculating class weights for imbalance...")
# Get class counts from the training dataset
counts = np.bincount([s[1] for s in dataloaders['train'].dataset])
# Calculate weights
weights = 1. / torch.tensor(counts, dtype=torch.float)
weights = weights.to(device)
print(f"Weights calculated: {weights.cpu().numpy()}")

# --- 3. Prepare the Model (SqueezeNet) ---
print("Preparing SqueezeNet model...")
model = models.squeezenet1_1(pretrained=True)

# Replace the final classifier for our 2-class problem
num_classes = 2
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model.num_classes = num_classes
model = model.to(device)

# --- 4. Train the Model ---
criterion = nn.CrossEntropyLoss(weight=weights) # Use the calculated weights
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 15

print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# --- 5. Evaluate with a Confusion Matrix ---
print("\nEvaluating model and plotting confusion matrix...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title('Confusion Matrix for Validation Data')
plt.show()

# =================================================================
# <<< NEW SECTION FOR GRAD-CAM ADDED HERE >>>
# =================================================================
# --- 6. Explain a Prediction with Grad-CAM ---
print("\nGenerating Grad-CAM visualization for a random validation image...")
model.eval()

# Define the target layer for SqueezeNet
target_layers = [model.features[-1]]

# Get a random image from the validation set
inputs, labels = next(iter(dataloaders['val']))
input_tensor = inputs[0].unsqueeze(0) # Get the first image of the batch
label = labels[0]
rgb_img = inputs[0].permute(1, 2, 0).numpy()
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img)) # Normalize to 0-1 for display

# Get the model's prediction for this image
output = model(input_tensor.to(device))
_, prediction_idx = torch.max(output, 1)
predicted_class = class_names[prediction_idx.item()]
true_class = class_names[label.item()]

# Create the Grad-CAM object and run it
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(prediction_idx.item())]
grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image in the batch

# Create the visualization by overlaying the heatmap
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Plot the result
plt.imshow(visualization)
plt.title(f'True: {true_class} | Predicted: {predicted_class}')
plt.axis('off')
plt.show()
