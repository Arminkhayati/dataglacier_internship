import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# read data
df = pd.read_csv('dataset/iris.data',names=['s_lenght', 's_width', 'p_length', 'p_width', 'iris_class'])
print(df.head())
# Feature matrix
X = df[['s_lenght', 's_width', 'p_length', 'p_width']].values
print('X shape = ', X.shape)
# Output variable
y = df[['iris_class']].values
print('Y shape = ', y.shape)
# Label encoder
encoder = LabelEncoder()
y = encoder.fit_transform(y.ravel())
joblib.dump(encoder, "saved_models/02.iris_label_encoder.pkl")
# split test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# train model
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
# Save Model
joblib.dump(classifier, "saved_models/01.knn_with_iris_dataset.pkl")

