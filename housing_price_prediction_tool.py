import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_columns = ['City', 'Neighborhood', 'Material']

VALID_CITIES = ['София', 'Пловдив', 'Плевен',]
TRAIN_MODEL = False

model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=150,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)


def prepare_data():
    data = pd.read_csv('data/all_data.csv')
    data = data.drop(columns=['Heating',])

    onehot_encoded = onehot_encoder.fit_transform(data[onehot_columns])
    onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(onehot_columns))

    data = data.drop(columns=onehot_columns)
    data = pd.concat([data, onehot_df], axis=1)

    X = data.drop(columns=['Price'])
    y = data['Price']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgb_regressor(X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    test_model(X_test, y_test)
    model.save_model('housing_price_model.json')
    return model


def test_model(X_test, y_test):
    y_pred = model.predict(X_test)
    y_test = y_test
    mae = mean_absolute_error(y_pred, y_test)
    print(mae)


def preprocess_input(input_data, X_train):
    onehot_columns = ['City', 'Neighborhood', 'Material']
    numerical_columns = ['Rooms', 'Size', 'Year', 'Total Floors', 'Floor']

    for col in onehot_columns:
        if col not in input_data:
            input_data[col] = 'Missing'
    for col in numerical_columns:
        if col not in input_data:
            input_data[col] = 0

    onehot_encoded = onehot_encoder.transform(input_data[onehot_columns])
    onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(onehot_columns))

    input_data = input_data.drop(columns=onehot_columns)
    input_data = pd.concat([input_data, onehot_df], axis=1)

    train_columns = X_train.columns.tolist()

    # Align the input data columns with the training data columns
    input_data = input_data[train_columns]

    return input_data


def predict(X_train):
    print('Valid cities ', VALID_CITIES)
    city = input('Enter City: ')
    while city not in VALID_CITIES:
        city = input('Enter City: ')
    city = f'град {city}'
    neighbourhood = input('Enter neighbourhood: ')
    neighbourhood = f'{neighbourhood} - {city}'
    rooms = int(input('Enter number of rooms: '))
    size = int(input('Enter apartment\'s size: '))
    total_floors = int(input('Enter total number of floors: '))
    floor = int(input('Enter floor: '))

    predict_data = pd.DataFrame([{
        'City': city,
        'Neighborhood': neighbourhood,
        'Rooms': rooms,
        'Size': size,
        'Total Floors': total_floors,
        'Floor': floor,
    }])
    year = input('Enter building year (if known): ')
    if year:
        predict_data['Year'] = int(year)

    building_material = input('Building material (if known): ')
    if building_material:
        predict_data['Material'] = building_material

    processed_input = preprocess_input(predict_data, X_train)

    prediction = model.predict(processed_input)

    return prediction


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    if TRAIN_MODEL:
        train_xgb_regressor(X_train, X_test, y_train, y_test)
    else:
        model.load_model('housing_price_model.json')
    while True:
        predicted_price = predict(X_train)
        print(f'Predicted Price: {predicted_price[0]:,.2f}')


if __name__ == '__main__':
    main()
