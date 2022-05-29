import pickle
import pandas as pd

model = pickle.load(open("data_analysis/price_prediction_model.pkl", "rb"))  # read mode


nndf = pd.read_csv("data_analysis/dataset/no_null_df.csv")  # non normalized dataframe
x_float_cols = [
    "Displacement(cc)",
    "Cylinders",
    "Fuel_Tank_Capacity(litres)",
    "Wheelbase",
    "Highway_Mileage(km/litre)",
    "Seating_Capacity",
    "Number_of_Airbags",
]


def normalize(input):
    for index, c in enumerate(x_float_cols):
        # print(index)
        input[index] = (input[index] - nndf[c].min()) / (nndf[c].max() - nndf[c].min())
    return input


input_cols = [1850, 4, 52, 2630, 20, 5, 2, True, True, True, True, True, True, True]
normalized_input = normalize(input_cols)
prediction = model.predict([normalized_input])
output = round(prediction[0], 2)
print(output)

input_cols = [
    1550,
    4,
    52,
    2630,
    20,
    5,
    2,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]
normalized_input = normalize(input_cols)
prediction = model.predict([normalized_input])
output = round(prediction[0], 2)
print(output)

# [0.3539697542533081, 0.3333333333333333, 0.5, 0.49640287769784175, 1.0, 0.5, 0.4, False, False, False, False, False, False, False]
