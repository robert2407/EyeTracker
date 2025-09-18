import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("iris_data_relative.csv")
print(f"Date citite: {len(df)} rânduri")

X = df[["lx_rel", "ly_rel", "rx_rel", "ry_rel"]].values
y = df["label"].values

scaler = StandardScaler() #normalizare
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

joblib.dump(knn, "knn_gaze_model.pkl")
joblib.dump(scaler, "scaler_gaze.pkl")
print("[INFO] Model salvat ca knn_gaze_model.pkl și scaler_gaze.pkl")
