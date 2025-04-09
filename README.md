# Cloud Seeding Vision AI

A computer vision system that analyzes cloud images to determine their suitability for cloud seeding operations. This project uses deep learning to classify clouds based on their visual characteristics and provides real-time analysis through a webcam interface.

## Features

- Cloud image classification (suitable vs. not suitable for seeding)
- Real-time webcam detection
- Automated image analysis and sorting
- Deep learning model with convolutional neural networks
- User-friendly command-line interface

## Requirements

- Python 3.8+
- OpenCV (cv2)
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Drone_P_A
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Drone_P_A/
├── Ai_vision_model/
│   └── cloudDataset/
│       └── sorted_clouds/
│           ├── suitable/
│           └── not_suitable/
├── cloud_model.py
├── cloud_analyzer.py
└── README.md
```

## Usage

### 1. Data Collection and Analysis

Run the cloud analyzer to process and sort cloud images:
```bash
python cloud_analyzer.py
```

### 2. Model Training

Train the AI model using the sorted images:
```bash
python cloud_model.py
```
Select option 1 from the menu to start training.

### 3. Real-time Detection

Use the trained model for live cloud analysis:
```bash
python cloud_model.py
```
Select option 2 from the menu to start live detection.

## Model Architecture

The model uses a convolutional neural network with the following structure:
- Input: 224x224x3 RGB images
- Convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers with ReLU activation
- Sigmoid output for binary classification

## Training Process

The model is trained with:
- Binary cross-entropy loss
- Adam optimizer
- Early stopping
- Learning rate reduction on plateau
- 20% validation split
- Batch size of 16

## Performance Metrics

The model provides:
- Accuracy
- Precision
- Recall
- Real-time confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Cloud Dataset
- NOAA Cloud Database
- TensorFlow and Keras communities
- OpenCV contributors

## Contact

For questions or suggestions, please open an issue in the repository. 