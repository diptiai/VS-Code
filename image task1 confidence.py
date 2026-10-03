import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# ============================================================
# 1. LOAD DIGIT DATASET
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset loaded successfully!")
print("Number of images:", X.shape[0])
print("Image size:", digits.images.shape[1], "x", digits.images.shape[2])


# ============================================================
# 2. NORMALIZE DATA
# ============================================================

X = X / 16.0

print("Data normalized between 0 and 1.")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

np.random.seed(42)

indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

def one_hot_encode(labels, num_classes=10):

    encoded = np.zeros((len(labels), num_classes))

    for i, label in enumerate(labels):
        encoded[i, label] = 1

    return encoded


y_train_onehot = one_hot_encode(y_train)


# ============================================================
# 5. INITIALIZE NEURAL NETWORK
# ============================================================

np.random.seed(42)

input_size = 64
hidden_size = 128
output_size = 10

W1 = np.random.randn(
    input_size,
    hidden_size
) * np.sqrt(2 / input_size)

b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(
    hidden_size,
    output_size
) * np.sqrt(2 / hidden_size)

b2 = np.zeros((1, output_size))


# ============================================================
# 6. ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):

    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-10

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        np.sum(
            y_true * np.log(y_pred),
            axis=1
        )
    )

    return loss


# ============================================================
# 8. FORWARD PROPAGATION
# ============================================================

def forward(X):

    # Input -> Hidden layer
    z1 = np.dot(X, W1) + b1

    # ReLU activation
    a1 = relu(z1)

    # Hidden -> Output layer
    z2 = np.dot(a1, W2) + b2

    # Softmax activation
    a2 = softmax(z2)

    return z1, a1, z2, a2


# ============================================================
# 9. TRAINING SETTINGS
# ============================================================

learning_rate = 0.05
epochs = 1000

loss_history = []
accuracy_history = []


# ============================================================
# 10. TRAIN THE NEURAL NETWORK
# ============================================================

print("\nStarting training...\n")

for epoch in range(epochs):

    # Forward propagation
    z1, a1, z2, predictions = forward(X_train)

    # Calculate loss
    loss = cross_entropy_loss(
        y_train_onehot,
        predictions
    )

    # Calculate accuracy
    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    accuracy = np.mean(
        predicted_classes == y_train
    )

    # Store values
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    m = X_train.shape[0]

    # Output layer gradient
    dz2 = predictions - y_train_onehot

    dW2 = np.dot(
        a1.T,
        dz2
    ) / m

    db2 = np.sum(
        dz2,
        axis=0,
        keepdims=True
    ) / m

    # Hidden layer gradient
    da1 = np.dot(
        dz2,
        W2.T
    )

    dz1 = (
        da1 *
        relu_derivative(z1)
    )

    dW1 = np.dot(
        X_train.T,
        dz1
    ) / m

    db1 = np.sum(
        dz1,
        axis=0,
        keepdims=True
    ) / m

    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Display training progress
    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )


# ============================================================
# 11. TEST THE MODEL
# ============================================================

_, _, _, test_predictions = forward(X_test)

test_predicted_classes = np.argmax(
    test_predictions,
    axis=1
)

test_accuracy = np.mean(
    test_predicted_classes == y_test
)

print("\n======================================")
print("MODEL TEST RESULTS")
print("======================================")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# 12. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.show()


# ============================================================
# 13. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    np.array(accuracy_history) * 100
)

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.show()


# ============================================================
# 14. DISPLAY TEST PREDICTIONS
# ============================================================

plt.figure(figsize=(10, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i].reshape(8, 8),
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {test_predicted_classes[i]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 15. USER IMAGE INPUT
# ============================================================

print("\n======================================")
print("USER IMAGE DIGIT CLASSIFICATION")
print("======================================")

image_path = input(
    "\nEnter the path of your digit image: "
)


# ============================================================
# 16. READ IMAGE
# ============================================================

try:

    img = plt.imread(image_path)

except Exception as e:

    print("\nError loading image.")
    print("Please check the image path.")
    print("Error:", e)

    exit()


# ============================================================
# 17. CONVERT IMAGE TO GRAYSCALE
# ============================================================

if img.ndim == 3:

    img = np.mean(
        img[:, :, :3],
        axis=2
    )


# ============================================================
# 18. CONVERT TO FLOAT
# ============================================================

img = img.astype(np.float32)


# ============================================================
# 19. NORMALIZE IMAGE
# ============================================================

if img.max() > 1:

    img = img / 255.0


# ============================================================
# 20. RESIZE IMAGE TO 8 x 8
# ============================================================

height, width = img.shape

row_indices = np.linspace(
    0,
    height - 1,
    8
).astype(int)

col_indices = np.linspace(
    0,
    width - 1,
    8
).astype(int)

img_8x8 = img[
    np.ix_(
        row_indices,
        col_indices
    )
]


# ============================================================
# 21. INVERT IMAGE IF NECESSARY
# ============================================================

if img_8x8.mean() > 0.5:

    img_8x8 = 1.0 - img_8x8


# ============================================================
# 22. DISPLAY NUMERICAL IMAGE
# ============================================================

print("\n======================================")
print("NUMERICAL IMAGE")
print("======================================")

print("\n8 x 8 normalized pixel values:\n")

print(
    np.round(
        img_8x8,
        2
    )
)


# ============================================================
# 23. CONVERT IMAGE INTO 64 INPUT VALUES
# ============================================================

image_input = img_8x8.reshape(
    1,
    64
)

print("\nFlattened image shape:")

print(image_input.shape)

print(
    "\nNumber of input pixels:",
    image_input.size
)


# ============================================================
# 24. PREDICT DIGIT
# ============================================================

_, _, _, probabilities = forward(
    image_input
)


# ============================================================
# 25. GET PREDICTED DIGIT
# ============================================================

predicted_digit = np.argmax(
    probabilities[0]
)


# ============================================================
# 26. GET CONFIDENCE
# ============================================================

confidence = (
    probabilities[0][predicted_digit]
    * 100
)


# ============================================================
# 27. DISPLAY PREDICTION AND CONFIDENCE
# ============================================================

print("\n======================================")
print("PREDICTION RESULT")
print("======================================")

print(
    "Predicted Digit:",
    predicted_digit
)

print(
    "Confidence: {:.2f}%".format(
        confidence
    )
)


# ============================================================
# 28. DISPLAY ALL CLASS PROBABILITIES
# ============================================================

print("\nClass probabilities:")

for digit in range(10):

    print(
        f"Digit {digit}: "
        f"{probabilities[0][digit] * 100:.2f}%"
    )


# ============================================================
# 29. DISPLAY PROCESSED USER IMAGE
# ============================================================

plt.figure(figsize=(5, 5))

plt.imshow(
    img_8x8,
    cmap="gray"
)

plt.title(
    f"Predicted Digit: {predicted_digit}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()

