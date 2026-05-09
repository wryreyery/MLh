import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# ১. Dataset
data = {
    'Age': [25, 30, 45, 50, 22, 35, 60, 55, 20, 40],
    'BMI': [22, 24, 30, 32, 21, 28, 35, 31, 19, 29],
    'Status': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
# ২. Features & Target
X = df[['Age', 'BMI']]
y = df['Status']
# ৩. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ৪. KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
# ৫. User Input
age = float(input("আপনার বয়স লিখুন: "))
bmi = float(input("আপনার BMI লিখুন: "))

new_person = np.array([[age, bmi]])
new_person_scaled = scaler.transform(new_person)
# Prediction
prediction = knn.predict(new_person_scaled)
# ৬. Result
print("\n===== Prediction Result =====")

if prediction[0] == 0:
    print("ফলাফল: Healthy (সুস্থ)")
else:
    print("ফলাফল: At Risk (স্বাস্থ্য ঝুঁকি আছে)")
# ৭. Graph
plt.figure(figsize=(8,6))
# Healthy মানুষ
plt.scatter(
    df[df['Status']==0]['Age'],
    df[df['Status']==0]['BMI'],
    color='green',
    s=120,
    label='Healthy'
)
# At Risk মানুষ
plt.scatter(
    df[df['Status']==1]['Age'],
    df[df['Status']==1]['BMI'],
    color='red',
    s=120,
    label='At Risk'
)
# New Person
plt.scatter(age,bmi,color='blue',s=250,marker='*',label='New Person')
# Labels
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("Health Risk Prediction")
plt.xticks(range(15, 70, 1))
plt.yticks(range(18, 40, 1))
plt.grid(True)
plt.legend()
plt.show()
