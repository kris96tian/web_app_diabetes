from flask import Flask, render_template, request
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Custom dataset class
class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural Network model
class DiabetesModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, criterion, optimizer, X_test_scaled, y_test, num_epochs=100):
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation Phase
    model.eval()
    with torch.no_grad():
        test_dataset = DiabetesDataset(X_test_scaled, y_test.values)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        for features, labels in test_loader:
            outputs = model(features)
            predicted = outputs.round()
            accuracy = (predicted == labels).float().mean()

data = pd.read_csv('diabetes_data.csv')
data.drop(['SkinThickness', 'DiabetesPedigreeFunction'], axis=1, inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.map({'alive': 0, 'dead': 1})
y_test = y_test.map({'alive': 0, 'dead': 1})

# SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Create datasets and data loaders
train_dataset = DiabetesDataset(X_train_resampled, y_train.values)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, criterion, and optimizer
input_size = X_train.shape[1]
model = DiabetesModel(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def diabetes_diagnosis(patient_data):
    train_and_evaluate(model, train_loader, criterion, optimizer, X_test, y_test, num_epochs=100)
    with torch.no_grad():
        # Ensure patient data is numeric
        patient_data = patient_data.apply(pd.to_numeric, errors='coerce')
        if patient_data.isnull().values.any():
            return "Error: Invalid input data. Please ensure all fields contain valid numeric values."

        patient_data_scaled = scaler.transform(patient_data)
        patient_tensor = torch.tensor(patient_data_scaled, dtype=torch.float32)
        prediction = model(patient_tensor)
        prediction_probability = prediction.item()

    if prediction_probability >= 0.5:
        result = f"High diabetes risk detected. Probability: {prediction_probability:.1%}. Please consult a healthcare professional."
    else:
        result = f"No diabetes risk detected. Probability: {prediction_probability:.1%}"
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = {
                'Pregnancies': [int(request.form['pregnancies'])],
                'Glucose': [int(request.form['glucose'])],
                'BloodPressure': [int(request.form['blood_pressure'])],
                'Insulin': [int(request.form['insulin'])],
                'BMI': [float(request.form['bmi'])],
                'Age': [int(request.form['age'])]
            }
            patient_data = pd.DataFrame(input_data)
            result = diabetes_diagnosis(patient_data)
        except ValueError:
            result = "Error: Please ensure all input fields contain valid numeric values."
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
