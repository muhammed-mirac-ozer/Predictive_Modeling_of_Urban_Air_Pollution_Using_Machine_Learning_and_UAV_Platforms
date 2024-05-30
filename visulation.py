import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_excel("sample_data.xlsx")

# Replace ',' with '.' and convert to numeric for relevant columns
data[['PM10', 'PM 2.5', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']] = data[['PM10', 'PM 2.5', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')

# Fill missing values with the nearest non-null value
data_filled = data.fillna(method='ffill')

# Save the filled data to a new Excel file
data_filled.to_excel("sample_data_filled.xlsx", index=False)

# Separate features and target variable
X = data_filled.drop(['Date', 'O3'], axis=1)
y = data_filled['O3']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing pipeline
model_pipeline = make_pipeline(
    StandardScaler()
)

# Fit and transform features
x_train_preprocessed = model_pipeline.fit_transform(x_train)
x_test_preprocessed = model_pipeline.transform(x_test)


# Veri setini yükle
data = pd.read_excel("sample_data_filled.xlsx")

# Korelasyon matrisini hesapla
correlation_matrix = data.corr()

# Korelasyon matrisi figürü
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

plt.show()

# Load dataset
data = pd.read_excel("sample_data_filled.xlsx")

# Tarih sütununu datetime türüne dönüştür
data['Date'] = pd.to_datetime(data['Date'])

# Veri setini indeks olarak Tarih sütununu ayarla
data.set_index('Date', inplace=True)

# Grafik boyutlarını belirle
plt.figure(figsize=(12, 12))

colors = ['blue', 'gold', 'green', 'red', 'purple', 'orange', 'forestgreen', 'peru']
# Her bir özellik için ayrı çizgi grafiği oluştur
features = ['PM10', 'PM 2.5', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']
for idx, feature in enumerate(features, start=1):
    plt.subplot(4, 2, idx)
    plt.plot(data.index, data[feature], color=colors[idx - 1])

    # Alt taraf için x eksenini göster ve kalın yap
    if idx > len(features) - 2:
        plt.xticks(rotation=45, ha='left', fontname='Times New Roman')
        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True,
                        width=2)
        plt.xlabel('Date', fontname='Times New Roman')
    else:
        plt.xticks([])

    plt.title(feature, fontname='Times New Roman')
    plt.ylabel('Values',  fontname='Times New Roman')

# Grafikleri göster
plt.tight_layout()
plt.show()
