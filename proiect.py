import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button

# Set the path to your dataset
dataset_path = './dataset'

# Function to load and resize images from a folder
def load_and_resize_images_from_folder(folder, target_size=(256, 256)):
    images = []
    labels = []
    class_name = os.path.basename(folder)  # Extract class name from folder path
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize the image to a consistent size
            images.append(img.flatten())
            labels.append(class_name)  # Assign only the class name
    return images, labels

# Load and resize images from the training set
train_images = []
train_labels = []

for flower_class in os.listdir(os.path.join(dataset_path, 'train')):
    folder_path = os.path.join(dataset_path, 'train', flower_class)
    images, labels = load_and_resize_images_from_folder(folder_path)
    train_images.extend(images)
    train_labels.extend(labels)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Create and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_images, train_labels)

# Create and train the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_images, train_labels)

# Create and train the Naive Bayes Classifier
nb = GaussianNB()
nb.fit(train_images, train_labels)

#Load and resize images from the validation set
validation_images = []
validation_labels = []

for flower_class in os.listdir(os.path.join(dataset_path, 'validation')):
    folder_path = os.path.join(dataset_path, 'validation', flower_class)
    images, labels = load_and_resize_images_from_folder(folder_path)
    validation_images.extend(images)
    validation_labels.extend(labels)
# Convert lists to numpy arrays
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

# Set the threshold for Naive Bayes
threshold_nb = 0.5 

# Make probability predictions on the validation set
rf_validation_pred_prob = rf.predict_proba(validation_images)
knn_validation_pred_prob = knn.predict_proba(validation_images)
nb_validation_pred_prob = nb.predict_proba(validation_images)

# Convert numeric labels back to class names
rf_predicted_class_names = rf.classes_[np.argmax(rf_validation_pred_prob, axis=1)]
knn_predicted_class_names = knn.classes_[np.argmax(knn_validation_pred_prob, axis=1)]
nb_predicted_class_names = nb.classes_[(nb_validation_pred_prob[:, 1] > threshold_nb).astype(int)]

# Load and resize images from the test set
test_images = []
test_labels = []

for flower_class in os.listdir(os.path.join(dataset_path, 'test')):
    folder_path = os.path.join(dataset_path, 'test', flower_class)
    images, labels = load_and_resize_images_from_folder(folder_path)
    test_images.extend(images)
    test_labels.extend(labels)

# Convert lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)


# Make probability predictions on the test set

rf_test_pred_prob = rf.predict_proba(test_images)
knn_test_pred_prob = knn.predict_proba(test_images)
nb_test_pred_prob = nb.predict_proba(test_images)

# Convert numeric labels back to class names
rf_test_predicted_class_names = rf.classes_[np.argmax(rf_test_pred_prob, axis=1)]
knn_test_predicted_class_names = knn.predict(test_images)
nb_test_predicted_class_names = nb.classes_[(nb_test_pred_prob[:, 1] > threshold_nb).astype(int)]

# Display and print results for the validation set
# Random Forest
rf_validation_pred_labels = rf.classes_[np.argmax(rf_validation_pred_prob, axis=1)]
rf_validation_accuracy = accuracy_score(validation_labels, rf_validation_pred_labels)
rf_validation_classification_rep = classification_report(validation_labels, rf_validation_pred_labels)
rf_validation_conf_matrix = confusion_matrix(validation_labels, rf_validation_pred_labels)

print(f"Random Forest Validation Accuracy: {rf_validation_accuracy:.2%}")
print("Random Forest Validation Classification Report:")
print(rf_validation_classification_rep)
print("Random Forest Validation Confusion Matrix:")
print(rf_validation_conf_matrix)

# KNN
knn_validation_accuracy = accuracy_score(validation_labels, knn_predicted_class_names)
knn_validation_classification_rep = classification_report(validation_labels, knn_predicted_class_names)
knn_validation_conf_matrix = confusion_matrix(validation_labels, knn_predicted_class_names)

print(f"KNN Validation Accuracy: {knn_validation_accuracy:.2%}")
print("KNN Validation Classification Report:")
print(knn_validation_classification_rep)
print("KNN Validation Confusion Matrix:")
print(knn_validation_conf_matrix)

#Naive Bayes
nb_validation_pred_labels = nb.classes_[(nb_validation_pred_prob[:, 1] > threshold_nb).astype(int)]
nb_validation_accuracy = accuracy_score(validation_labels, nb_validation_pred_labels)
nb_validation_classification_rep = classification_report(validation_labels, nb_validation_pred_labels)
nb_validation_conf_matrix = confusion_matrix(validation_labels, nb_validation_pred_labels)

print(f"Naive Bayes Validation Accuracy: {nb_validation_accuracy:.2%}")
print("Naive Bayes Validation Classification Report:")
print(nb_validation_classification_rep)
print("Naive Bayes Validation Confusion Matrix:")
print(nb_validation_conf_matrix)

# Display and print results for the test set
#Random Forest
rf_test_pred_labels = rf.classes_[np.argmax(rf_test_pred_prob, axis=1)]
rf_test_accuracy = accuracy_score(test_labels, rf_test_pred_labels)
rf_test_classification_rep = classification_report(test_labels, rf_test_pred_labels)
rf_test_conf_matrix = confusion_matrix(test_labels, rf_test_pred_labels)

print(f"\nRandom Forest Test Accuracy: {rf_test_accuracy:.2%}")
print("Random Forest Test Classification Report:")
print(rf_test_classification_rep)
print("Random Forest Test Confusion Matrix:")
print(rf_test_conf_matrix)

# KNN
knn_test_accuracy = accuracy_score(test_labels, knn_test_predicted_class_names)
knn_test_classification_rep = classification_report(test_labels, knn_test_predicted_class_names)
knn_test_conf_matrix = confusion_matrix(test_labels, knn_test_predicted_class_names)

print(f"\nKNN Test Accuracy: {knn_test_accuracy:.2%}")
print("KNN Test Classification Report:")
print(knn_test_classification_rep)
print("KNN Test Confusion Matrix:")
print(knn_test_conf_matrix)

#Naive Bayes
nb_test_pred_labels = nb.classes_[(nb_test_pred_prob[:, 1] > threshold_nb).astype(int)]
nb_test_accuracy = accuracy_score(test_labels, nb_test_pred_labels)
nb_test_classification_rep = classification_report(test_labels, nb_test_pred_labels)
nb_test_conf_matrix = confusion_matrix(test_labels, nb_test_pred_labels)

print(f"\nNaive Bayes Test Accuracy: {nb_test_accuracy:.2%}")
print("Naive Bayes Test Classification Report:")
print(nb_test_classification_rep)
print("Naive Bayes Test Confusion Matrix:")
print(nb_test_conf_matrix)


class ImageViewer:
    def __init__(self, images, true_labels, predicted_probabilities_1, class_names_1, title_1, predicted_probabilities_2, class_names_2, title_2, predicted_probabilities_3, class_names_3, title_3):
        self.images = images
        self.true_labels = true_labels

        self.predicted_probabilities_1 = predicted_probabilities_1
        self.class_names_1 = class_names_1
        self.title_1 = title_1

        self.predicted_probabilities_2 = predicted_probabilities_2
        self.class_names_2 = class_names_2
        self.title_2 = title_2

        self.predicted_probabilities_3 = predicted_probabilities_3
        self.class_names_3 = class_names_3
        self.title_3 = title_3

        self.index = 0

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.display_images()

        self.next_button = Button(plt.axes([0.7, 0.02, 0.1, 0.05]), 'Next')
        self.next_button.on_clicked(self.next_image)

        plt.show()

    def display_images(self):
        img = self.images[self.index].reshape(256, 256, 3)
        true_class = self.true_labels[self.index]
        
        # Display predictions for the first classifier
        predicted_probs_1 = self.predicted_probabilities_1[self.index]
        top_class_index_1 = np.argmax(predicted_probs_1)
        top_class_name_1 = self.class_names_1[top_class_index_1]
        top_class_prob_1 = predicted_probs_1[top_class_index_1]

        # Display predictions for the second classifier
        predicted_probs_2 = self.predicted_probabilities_2[self.index]
        top_class_index_2 = np.argmax(predicted_probs_2)
        top_class_name_2 = self.class_names_2[top_class_index_2]
        top_class_prob_2 = predicted_probs_2[top_class_index_2]

        # Display predictions for the third classifier
        predicted_probs_3 = self.predicted_probabilities_3[self.index]
        top_class_index_3 = np.argmax(predicted_probs_3)
        top_class_name_3 = self.class_names_3[top_class_index_3]
        top_class_prob_3 = predicted_probs_3[top_class_index_3]


        # Display the first image with predictions
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.ax1.set_title(f"{self.title_1}\nTrue: {true_class}\nPredicted: {top_class_name_1} ({top_class_prob_1:.2%})")
        self.ax1.axis('off')

        # Display the second image with predictions
        self.ax2.clear()
        self.ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.ax2.set_title(f"{self.title_2}\nTrue: {true_class}\nPredicted: {top_class_name_2} ({top_class_prob_2:.2%})")
        self.ax2.axis('off')

        # Display the second image with predictions
        self.ax3.clear()
        self.ax3.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.ax3.set_title(f"{self.title_3}\nTrue: {true_class}\nPredicted: {top_class_name_3} ({top_class_prob_3:.2%})")
        self.ax3.axis('off')

        plt.draw()

    def next_image(self, event):
        self.index = (self.index + 1) % len(self.images)
        self.display_images()
        plt.draw()

# Initialize the image viewer for the test set with Random Forest and Naive Bayes
test_image_viewer = ImageViewer(test_images, test_labels, rf_test_pred_prob, rf.classes_, "Random Forest", nb_test_pred_prob, nb.classes_, "Naive Bayes", knn_test_pred_prob, knn.classes_, "K-Nearest Neighbours")

# Show the figure
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(12, 18))
plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing

# Random Forest
sns.heatmap(rf_validation_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_, ax=axes[0, 0])
axes[0, 0].set_title('Random Forest Validation Confusion Matrix')

sns.heatmap(rf_test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_, ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Test Confusion Matrix')

# Naive Bayes
sns.heatmap(nb_validation_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=nb.classes_, yticklabels=nb.classes_, ax=axes[1, 0])
axes[1, 0].set_title('Naive Bayes Validation Confusion Matrix')

sns.heatmap(nb_test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=nb.classes_, yticklabels=nb.classes_, ax=axes[1, 1])
axes[1, 1].set_title('Naive Bayes Test Confusion Matrix')

# KNN
sns.heatmap(knn_validation_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_, ax=axes[2, 0])
axes[2, 0].set_title('KNN Validation Confusion Matrix')

sns.heatmap(knn_test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_, ax=axes[2, 1])
axes[2, 1].set_title('KNN Test Confusion Matrix')

plt.show()

classifiers = ['Random Forest', 'Naive Bayes', 'K-Nearest Neighbours']
validation_accuracies = [rf_validation_accuracy, nb_validation_accuracy, knn_validation_accuracy]
test_accuracies = [rf_test_accuracy, nb_test_accuracy, knn_test_accuracy]

bar_width = 0.35
index = np.arange(len(classifiers))

fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(index, validation_accuracies, bar_width, label='Validation Accuracy')
bar2 = ax.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy')

ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Validation and Test Accuracy for Each Classifier')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(classifiers)
ax.legend()

plt.show()