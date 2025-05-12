# Mengimport packages
import pandas as pd

# Mengimport classes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Mengimport function
from sklearn.model_selection import train_test_split


def read_dataset(filename, index_column, drop_column, label_column):
    """
    Membaca file dataset, menentukan index DataFrame, menghapus kolom yang belum diperlukan, dan mapping nilai kolom label

    Parameter:
        filename (str): nama file dari raw dataset
        index_column (str): nama kolom yang menjadi index DataFrame
        drop_column (str): nama kolom yang belum diperlukan
        label_column (str): nama kolom yang menjadi label

    Return:
        encoded_df (DataFrame): DataFrame yang telah melalui proses dalam function ini
    """
    # Membaca file dataset dan menentukan index DataFrame: df
    df = pd.read_csv(filename, index_col=index_column)

    # Menghapus kolom yang belum diperlukan
    df.drop(columns=drop_column, inplace=True)

    # Mapping nilai kolom label
    df[label_column] = df[label_column].map({"Yes": 1, "No": 0})

    return df


def label_based_encode_categorical_columns(df, categorical_columns, label_column):
    """
    Encoding data kategorik berdasarkan kolom label

    Parameter:
        df (DataFrame): DataFrame yang telah melalui proses dalam function read_dataset()
        categorical_columns (List[str]): daftar kolom kategorik
        label_column (str): nama kolom yang menjadi label

    Return:
        encoded_df (DataFrame): DataFrame yang telah melalui proces dalam function ini
    """
    # Copy DataFrame: encoded_df
    encoded_df = df.copy()

    # Encoding data kategorik berdasarkan kolom label
    for col in categorical_columns:
        encoding_map = encoded_df.groupby(col)[label_column].mean().to_dict()
        encoded_df[col] = encoded_df[col].map(encoding_map)

    return encoded_df


def handle_missing_values_and_standardize_features(df, label_column):
    """
    Menangani missing values dan standarisasi semua fitur

    Parameter:
        df (DataFrame): DataFrame yang telah melalui proses dalam function label_based_encode_categorical_columns()
        label_column (str): nama kolom yang menjadi label

    Return:
        transform_df (DataFrame): DataFrame yang telah melalui proces dalam function ini
    """
    # Fitur yang akan digunakan: features
    features = df.drop(columns=label_column, axis=1)

    # Object SimpleImputer untuk menangani missing values: imputer
    imputer = SimpleImputer(strategy="mean")

    # Menangani missing values
    impute_missing_values = imputer.fit_transform(features.values)
    imputed_features = pd.DataFrame(impute_missing_values, columns=features.columns)

    # Standarisasi semua fitur
    scaler = StandardScaler()
    standardize_features = scaler.fit_transform(imputed_features.values)
    standardized_features = pd.DataFrame(standardize_features, columns=features.columns)

    # Menggabungkan fitur dengan label
    transform_df = pd.concat([standardized_features, df[label_column]], axis=1)

    return transform_df


def split_dataset(df, label_column):
    """
    Membagi dataset pada fitur dan kolom menjadi masing-masing train set dan test set

    Parameter:
        df (DataFrame): DataFrame yang telah melalui proses dalam function label_based_encode_categorical_columns()
        label_column (str): nama kolom yang menjadi label
    Return:
        X_train (DataFrame): fitur untuk train set
        X_test (DataFrame): fitur untuk test set
        y_train (Series): label untuk train set
        y_test (Series): label untuk test set
    """
    # Membagi dataset menjadi fitur dan kolom: X dan y
    X = df.drop(columns=label_column, axis=1)
    y = df[label_column]

    # Membagi fitur dan kolom menjadi masing-masing train set dan test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1993
    )

    return X_train, X_test, y_train, y_test


def main():
    """
    Menjalankan semua function untuk data preprocessing dan menyimpan hasil data preprocessing

    Parameter: -

    Return: -
    """
    # Kolom label: label_column
    label_column = "RainTomorrow"

    # 1. Membaca file dataset, menentukan index DataFrame, menghapus kolom yang belum diperlukan, dan mapping nilai kolom label
    weather = read_dataset(
        filename="weather_raw.csv",
        index_column="Unnamed: 0",
        drop_column="Date",
        label_column=label_column,
    )

    # 2. Encoding data kategorik berdasarkan kolom label
    categorical_columns = weather.select_dtypes(include="object").columns
    weather = label_based_encode_categorical_columns(
        df=weather, categorical_columns=categorical_columns, label_column=label_column
    )

    # 3. Menangani missing values dan standarisasi semua fitur
    weather = handle_missing_values_and_standardize_features(
        df=weather, label_column=label_column
    )

    # 4. Membagi dataset pada fitur dan kolom menjadi masing-masing train set dan test set
    X_train, X_test, y_train, y_test = split_dataset(
        df=weather, label_column=label_column
    )

    # 5. Menyimpan hasil data preprocessing
    X_train.to_csv("preprocessing/weather_preprocessing/X_train.csv", index=False)
    X_test.to_csv("preprocessing/weather_preprocessing/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["RainTomorrow"]).to_csv(
        "preprocessing/weather_preprocessing/y_train.csv", index=False
    )
    pd.DataFrame(y_test, columns=["RainTomorrow"]).to_csv(
        "preprocessing/weather_preprocessing/y_test.csv", index=False
    )


if __name__ == "__main__":
    # Menjalankan function main()
    main()
