import numpy as np
import matplotlib.pyplot as plt

# Dataset is used only for training/testing data.
# The neural network itself is implemented completely with NumPy.
from sklearn.datasets import load_digits


# ============================================================
# 1. LOAD DIGIT DATASET
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

print("Original dataset shape:", X.shape)
print("Labels shape:", y.shape)


# ============================================================
# 2. VISUALIZE SOME ORIGINAL DIGITS
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        digits.images[i],
        cmap="gray"
    )

    plt.title(f"Digit: {y[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 3. NORMALIZE DATA
# ============================================================

# Original pixel values are between 0 and 16.
# Convert them to values between 0 and 1.

X = X / 16.0


# ============================================================
# 4. SHUFFLE DATA
# ============================================================

np.random.seed(42)

indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

def one_hot(y, num_classes=10):

    result = np.zeros(
        (len(y), num_classes)
    )

    result[
        np.arange(len(y)),
        y
    ] = 1

    return result


Y_train = one_hot(y_train)


# ============================================================
# 7. RELU ACTIVATION FUNCTION
# ============================================================

def relu(x):

    return np.maximum(0, x)


# ============================================================
# 8. SOFTMAX ACTIVATION FUNCTION
# ============================================================

def softmax(x):

    # Numerical stability
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
# 9. INITIALIZE NEURAL NETWORK
# ============================================================

input_size = 64
hidden_size = 128
output_size = 10

np.random.seed(42)

# Input → Hidden
W1 = (
    np.random.randn(
        input_size,
        hidden_size
    )
    * np.sqrt(2 / input_size)
)

b1 = np.zeros(
    (1, hidden_size)
)


# Hidden → Output
W2 = (
    np.random.randn(
        hidden_size,
        output_size
    )
    * np.sqrt(2 / hidden_size)
)

b2 = np.zeros(
    (1, output_size)
)


print("\nNetwork architecture:")
print("Input layer :", input_size)
print("Hidden layer:", hidden_size)
print("Output layer:", output_size)


# ============================================================
# 10. FORWARD PROPAGATION
# ============================================================

def forward(X):

    # Input → Hidden
    z1 = X @ W1 + b1

    # ReLU
    a1 = relu(z1)

    # Hidden → Output
    z2 = a1 @ W2 + b2

    # Softmax
    a2 = softmax(z2)

    return z1, a1, z2, a2


# ============================================================
# 11. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
    y_true,
    y_pred
):

    epsilon = 1e-9

    y_pred = np.clip(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = -np.sum(
        y_true * np.log(y_pred)
    ) / len(y_true)

    return loss


# ============================================================
# 12. ACCURACY FUNCTION
# ============================================================

def accuracy(
    y_true,
    y_pred
):

    predicted_labels = np.argmax(
        y_pred,
        axis=1
    )

    return np.mean(
        predicted_labels == y_true
    )


# ============================================================
# 13. TRAINING SETTINGS
# ============================================================

learning_rate = 0.1
epochs = 100

loss_history = []
accuracy_history = []


# ============================================================
# 14. TRAIN THE NEURAL NETWORK
# ============================================================

print("\nStarting training...\n")

for epoch in range(epochs):


    # --------------------------------------------------------
    # FORWARD PROPAGATION
    # --------------------------------------------------------

    z1, a1, z2, predictions = forward(
        X_train
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = cross_entropy_loss(
        Y_train,
        predictions
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    acc = accuracy(
        y_train,
        predictions
    )


    loss_history.append(loss)
    accuracy_history.append(acc)


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    m = len(X_train)


    # Output layer gradient
    dz2 = (
        predictions - Y_train
    ) / m


    # Gradient of W2
    dW2 = a1.T @ dz2


    # Gradient of b2
    db2 = np.sum(
        dz2,
        axis=0,
        keepdims=True
    )


    # Gradient flowing into hidden layer
    da1 = dz2 @ W2.T


    # ReLU derivative
    dz1 = da1 * (z1 > 0)


    # Gradient of W1
    dW1 = X_train.T @ dz1


    # Gradient of b1
    db1 = np.sum(
        dz1,
        axis=0,
        keepdims=True
    )


    # --------------------------------------------------------
    # UPDATE WEIGHTS AND BIASES
    # --------------------------------------------------------

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


    # --------------------------------------------------------
    # PRINT TRAINING PROGRESS
    # --------------------------------------------------------

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {acc * 100:.2f}%"
        )


# ============================================================
# 15. TEST THE NETWORK
# ============================================================

_, _, _, test_predictions = forward(
    X_test
)

test_accuracy = accuracy(
    y_test,
    test_predictions
)

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 16. VISUALIZE TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    loss_history
)

plt.title(
    "Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Cross-Entropy Loss"
)

plt.grid(True)

plt.show()


# ============================================================
# 17. VISUALIZE TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    accuracy_history
)

plt.title(
    "Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.grid(True)

plt.show()


# ============================================================
# 18. PREDICT TEST IMAGES
# ============================================================

predicted_labels = np.argmax(
    test_predictions,
    axis=1
)


plt.figure(figsize=(10, 7))

for i in range(12):

    plt.subplot(
        3,
        4,
        i + 1
    )

    image = X_test[i].reshape(
        8,
        8
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 19. INPUT IMAGE → NUMERICAL VALUES
# ============================================================

def preprocess_input_image(
    image_path
):

    """
    Converts a user's digit image into
    numerical pixel values.

    Steps:

    Image
       ↓
    Grayscale
       ↓
    Normalize
       ↓
    Resize to 8x8
       ↓
    Invert if necessary
       ↓
    Flatten
       ↓
    64 numerical values
    """


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = plt.imread(
        image_path
    )


    # --------------------------------------------------------
    # CONVERT RGB/RGBA TO GRAYSCALE
    # --------------------------------------------------------

    if image.ndim == 3:

        # Ignore alpha channel if present
        rgb = image[:, :, :3]

        image = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )


    # --------------------------------------------------------
    # CONVERT TO FLOAT
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )


    # --------------------------------------------------------
    # NORMALIZE PIXEL VALUES
    # --------------------------------------------------------

    if image.max() > 1:

        image = image / 255.0


    # --------------------------------------------------------
    # RESIZE IMAGE TO 8 × 8
    # --------------------------------------------------------

    height, width = image.shape

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

    image = image[
        np.ix_(
            row_indices,
            col_indices
        )
    ]


    # --------------------------------------------------------
    # INVERT IMAGE IF BACKGROUND IS WHITE
    # --------------------------------------------------------

    if image.mean() > 0.5:

        image = 1.0 - image


    # --------------------------------------------------------
    # CLIP VALUES
    # --------------------------------------------------------

    image = np.clip(
        image,
        0,
        1
    )


    # --------------------------------------------------------
    # DISPLAY NUMERICAL 8 × 8 MATRIX
    # --------------------------------------------------------

    print("\n===================================")
    print("8 × 8 NUMERICAL IMAGE")
    print("===================================")

    print(
        np.round(
            image,
            2
        )
    )


    # --------------------------------------------------------
    # FLATTEN IMAGE
    # --------------------------------------------------------

    numerical_input = image.reshape(
        1,
        64
    )


    print("\n===================================")
    print("64 NUMERICAL INPUT VALUES")
    print("===================================")

    print(
        np.round(
            numerical_input,
            2
        )
    )


    print(
        "\nInput shape:",
        numerical_input.shape
    )


    return (
        numerical_input,
        image
    )


# ============================================================
# 20. PREDICT USER'S IMAGE
# ============================================================

# Change this to your image filename.
#
# Example:
# image_path = "digit7.png"

image_path = "my_digit.png"


try:

    input_pixels, processed_image = (
        preprocess_input_image(
            image_path
        )
    )


    # --------------------------------------------------------
    # SEND IMAGE THROUGH NEURAL NETWORK
    # --------------------------------------------------------

    _, _, _, prediction = forward(
        input_pixels
    )


    # --------------------------------------------------------
    # GET PREDICTED DIGIT
    # --------------------------------------------------------

    predicted_digit = np.argmax(
        prediction
    )


    # --------------------------------------------------------
    # GET CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        np.max(prediction) * 100
    )


    # --------------------------------------------------------
    # DISPLAY ALL CLASS PROBABILITIES
    # --------------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "DIGIT PROBABILITIES"
    )

    print(
        "==================================="
    )


    for digit in range(10):

        probability = (
            prediction[0][digit] * 100
        )

        print(
            f"Digit {digit}: "
            f"{probability:.2f}%"
        )


    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        f"PREDICTED DIGIT: {predicted_digit}"
    )

    print(
        f"CONFIDENCE: {confidence:.2f}%"
    )

    print(
        "==================================="
    )


    # --------------------------------------------------------
    # DISPLAY PROCESSED INPUT IMAGE
    # --------------------------------------------------------

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        processed_image,
        cmap="gray"
    )

    plt.title(
        f"Predicted Digit: "
        f"{predicted_digit}\n"
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    plt.axis("off")

    plt.show()


except FileNotFoundError:

    print(
        "\nInput image not found."
    )

    print(
        "Place your digit image in the "
        "same folder as this Python file."
    )

    print(
        "Then change:"
    )

    print(
        'image_path = "my_digit.png"'
    )

    print(
        "to your actual image filename."
    )