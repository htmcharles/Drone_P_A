import cv2
import numpy as np
import os
import datetime
from tensorflow.keras import layers, models
import tensorflow as tf

# Create necessary directories
def create_data_folders():
    os.makedirs("Ai_vision_model/cloudDataset/sorted_clouds/suitable", exist_ok=True)
    os.makedirs("Ai_vision_model/cloudDataset/sorted_clouds/not_suitable", exist_ok=True)

def prepare_data():
    images = []
    labels = []
    
    # Load suitable images (label 1)
    suitable_dir = "Ai_vision_model/cloudDataset/sorted_clouds/suitable"
    print(f"Loading suitable images from {suitable_dir}...")
    for img_name in os.listdir(suitable_dir):
        img_path = os.path.join(suitable_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(1)
    print(f"Loaded {len([l for l in labels if l == 1])} suitable images")
    
    # Load not suitable images (label 0)
    not_suitable_dir = "Ai_vision_model/cloudDataset/sorted_clouds/not_suitable"
    print(f"Loading not suitable images from {not_suitable_dir}...")
    for img_name in os.listdir(not_suitable_dir):
        img_path = os.path.join(not_suitable_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(0)
    print(f"Loaded {len([l for l in labels if l == 0])} not suitable images")
    
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def run_live_detection():
    # Load the trained model
    if not os.path.exists('cloud_model.h5'):
        print("Error: No trained model found. Please train the model first.")
        return
    
    model = models.load_model('cloud_model.h5')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting live detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Preprocess frame
        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = processed_frame / 255.0
        processed_frame = np.expand_dims(processed_frame, axis=0)
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)[0][0]
        
        # Add prediction text to frame
        text = f"Suitable: {prediction:.2%}"
        color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display frame
        cv2.imshow('Cloud Seeding Detection', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n=== Cloud Seeding Vision AI ===")
        print("1. Train Model")
        print("2. Run Live Detection")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            print("Loading and preparing data...")
            X, y = prepare_data()
            
            if len(X) < 10:
                print("Error: Not enough images for training. Need at least 10 images.")
                continue
            
            print("Creating and training model...")
            model = create_model()
            
            # Compile model with appropriate metrics
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Add callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=0.00001
                )
            ]
            
            # Train the model
            history = model.fit(
                X, y,
                epochs=50,
                validation_split=0.2,
                batch_size=16,  # Reduced batch size
                callbacks=callbacks
            )
            
            # Save the model
            model.save('cloud_model.h5')
            print("Model trained and saved successfully!")
            
        elif choice == '2':
            run_live_detection()
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 