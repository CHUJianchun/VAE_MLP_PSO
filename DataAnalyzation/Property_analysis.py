import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd


def collect_from_dataset():
    with open('Data/data_capacity.data', 'rb') as f:
        data_h = pickle.load(f)
    with open('Data/data_conductivity.data', 'rb') as f:
        data_t = pickle.load(f)

    def regress_function(x, a, b):
        return a * x + b

    def collection(data):
        list_303 = []
        list_363 = []
        for item in data:
            temperature_index = None
            pressure_index = None
            property_index = None

            if item['title'] == 'Transport properties: Thermal conductivity':
                property_str = 'Thermal conductivity'
            elif item['title'] == 'Heat capacity and derived properties: Heat capacity at constant pressure':
                property_str = 'Heat capacity at constant pressure'

            for index in range(len(item['dhead'])):
                if 'Temperature' in item['dhead'][index][0]:
                    temperature_index = index
                elif 'Pressure' in item['dhead'][index][0]:
                    pressure_index = index
                elif property_str in item['dhead'][index][0]:
                    property_index = index

            try:
                assert isinstance(temperature_index, int)
                assert isinstance(property_index, int)
            except AssertionError:
                print(item['title'], item['dhead'])

            if not item['dhead'][property_index][1] == 'Liquid':
                print('Warning: No. ' + str(i) + ': The phase of this data series is '
                      + item['dhead'][property_index][1] + ', pass')
                continue
            t_value = np.zeros((len(item['data']), 2))

            for i in range(len(item['data'])):  # data_point -> list: 3
                data_point = item['data'][i]
                data_point_temperature = data_point[temperature_index][0]
                if pressure_index is None:
                    data_point_pressure = 101.325
                else:
                    data_point_pressure = data_point[pressure_index][0]
                data_point_value = data_point[property_index][0]
                if not 99. < float(data_point_pressure) < 105.:
                    continue
                t_value[i, 0] = data_point_temperature
                t_value[i, 1] = data_point_value
            if np.min(np.abs(t_value[:, 0] - 303)) > 5 or np.min(np.abs(t_value[:, 0] - 303)) > 5:
                continue
            try:
                a_, b_ = optimize.curve_fit(regress_function, t_value[:, 0].tolist(), t_value[:, 1].tolist())[0]
            except TypeError:
                print(t_value.shape)
                continue
            property_303 = regress_function(303, a_, b_)
            property_363 = regress_function(363, a_, b_)
            list_303.append(property_303)
            list_363.append(property_363)
        return list_303, list_363

    h_collection_303_, h_collection_363_ = collection(data_h)
    t_collection_303_, t_collection_363_ = collection(data_t)

    pd_data_ = pd.DataFrame(h_collection_303_)
    pd_data_.to_csv('h_303.csv')
    pd_data_ = pd.DataFrame(h_collection_363_)
    pd_data_.to_csv('h_363.csv')
    pd_data_ = pd.DataFrame(t_collection_363_)
    pd_data_.to_csv('t_363.csv')
    pd_data_ = pd.DataFrame(t_collection_303_)
    pd_data_.to_csv('t_303.csv')

    return np.array(h_collection_303_), \
           np.array(h_collection_363_), \
           np.array(t_collection_303_), \
           np.array(t_collection_363_)


if __name__ == '__main__':
    h_collection_303, h_collection_363, t_collection_303, t_collection_363 = collect_from_dataset()
    pd_data = pd.DataFrame(h_collection_303)
    pd_data.to_csv('h_303.csv')
    pd_data = pd.DataFrame(h_collection_363)
    pd_data.to_csv('h_363.csv')
    pd_data = pd.DataFrame(t_collection_363)
    pd_data.to_csv('t_363.csv')
    pd_data = pd.DataFrame(t_collection_303)
    pd_data.to_csv('t_303.csv')
