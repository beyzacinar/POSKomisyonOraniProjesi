import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error
)

import warnings

# --- 1. AYARLAR VE VERİ YÜKLEME ---
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

file_name = "Book4.xlsx"

try:
    # Kendi bilgisayarındaki dosya yoluna göre burayı düzenleyebilirsin
    path = r"C:\Users\ehayek\Desktop\Book4.xlsx"

    df = pd.read_excel(path)

    print(f"Veri başarıyla yüklendi! Satır sayısı: {len(df)}")

except Exception as e:
    print(f"HATA: Dosya okunamadı. Detay: {e}")

# --- 2. GÜÇLENDİRİLMİŞ VERİ ÖN İŞLEME ---

# A. Gereksiz sütunları çıkar
drop_cols = ['ÜyeNo', 'Müşteri No', 'TAX_OFFICE_NAME', 'Dönem']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# B. Akıllı Eksik Değer Doldurma
for col in df.columns:

    # Eğer sütun sayısal ise (int veya float) medyan ile doldur
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    # Eğer sütun yazı/kategori ise en çok tekrar eden (mode) doldur
    else:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

# C. Kategorik Değişkenleri Sayısallaştırma
le = LabelEncoder()

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = le.fit_transform(df[col].astype(str))

# --- 3. MODEL KURULUMU VE EĞİTİM ---

# Hedef değişken 'Debit Oranı'
X = df.drop('Debit Oranı', axis=1)
y = df['Debit Oranı']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Model eğitiliyor (Bu işlem veri setine göre 10-30 sn sürebilir)...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- 4. SONUÇLAR VE HATA METRİKLERİ ---

# Metrik Hesaplamaları
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE, MSE'nin kareköküdür
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n--- MODEL PERFORMANSI ---")
print(f"MAE (Ortalama Mutlak Hata): {mae:.4f}")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.4f}")
print(f"MAPE (Ortalama Mutlak Yüzde Hata): %{mape * 100:.2f}")
print(f"R2 Skoru (Açıklayıcılık Oranı): {r2:.4f}")

# --- 5. GRAFİK KAYDETME ---

plt.figure(figsize=(10, 8))

feat_importances = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feat_importances.nlargest(10).sort_values().plot(
    kind="barh",
    color="darkblue"
)

plt.title("Debit Oranını Etkileyen En Önemli 10 Faktör")
plt.tight_layout()

plt.savefig("özellik_onemi.png")

print("İşlem başarıyla tamamlandı. Grafik 'özellik_onemi.png' olarak kaydedildi.")

plt.show()