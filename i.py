from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Simple dataset: [height, weight]
X = [[150, 50],
     [160, 60],
     [170, 70],
     [180, 80]]
y = [0, 0, 1, 1]  # Labels: 0 = Short, 1 = Tall

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for the same data (for demonstration)
y_pred = model.predict(X)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)