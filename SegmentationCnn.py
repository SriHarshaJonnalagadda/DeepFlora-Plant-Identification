import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

# Define the paths and constants
base_dir = r'C:\Users\harsh\OneDrive\Documents\PythonLearn\Medicinal Leaf Dataset'
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,  # Add rotation to data augmentation
    brightness_range=[0.2, 1.2],  # Adjust brightness
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    horizontal_flip=True,
    vertical_flip=True,  # Add vertical flip
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

validation_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Create a pre-trained MobileNetV2 model for feature extraction n 
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(89, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the last layers for fine-tuning
for layer in base_model.layers:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement learning rate scheduling
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=20,  # Increase the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[reduce_lr]  # Add the learning rate scheduler
)

# Save the trained model
model.save('Improved_Hackathon_cnn.keras')
