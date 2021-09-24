from DataPreparation.IonicLiquid import *
import pickle
import importlib
from DataPreparation.Read_data import read_3_properties_data_files


class DataPoint:

    list_of_all_Data_Points = []

    def __init__(self, ionic_liquid_, temperature_, pressure_, property_, value_):
        self.ionic_liquid = ionic_liquid_
        self.temperature = temperature_
        self.pressure = pressure_
        self.property = property_
        self.value = value_
        DataPoint.list_of_all_Data_Points.append(self)


def create_data_point():
    # importlib.reload(IonicLiquid)
    data_viscosity, data_conductivity, data_capacity = read_3_properties_data_files()
    list_ = data_viscosity + data_conductivity + data_capacity
    str_viscosity = 'Transport properties: Viscosity'
    str_conductivity = 'Transport properties: Thermal conductivity'
    str_capacity = 'Heat capacity and derived properties: Heat capacity at constant pressure'
    try:
        with open('Data/ionic_liquid_list.data', 'rb') as f_ionic_liquid_list:
            IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    except IOError:
        print('Warning: File Data/ionic_liquid_list.data not found, creating')
        create_ionic_liquid()
        IonicLiquid.list_of_all_Ionic_Liquids = pickle.load(f_ionic_liquid_list)
    print('Start: data points list creating')
    iter_ = 0
    for i in range(len(list_)):
        item = list_[i]
        # 确定离子液体
        data_point_ionic_liquid = 1
        for ionic_liquid in IonicLiquid.list_of_all_Ionic_Liquids:
            if item['components'][0]['name'] == ionic_liquid.name:
                data_point_ionic_liquid = ionic_liquid
                break
        try:
            assert isinstance(data_point_ionic_liquid, IonicLiquid)
        except AssertionError:
            print(' Warning: No. ' + str(i) + ' Ionic Liquid ' + item['components'][0]['name'] +
                  ' not found, continue to next data frame')
            continue
        # 确定物性
        if item['title'] == str_viscosity:
            data_point_property = 'viscosity'
            property_str = 'Viscosity'
        elif item['title'] == str_conductivity:
            data_point_property = 'thermal_conductivity'
            property_str = 'Thermal conductivity'
        elif item['title'] == str_capacity:
            data_point_property = 'heat_capacity'
            property_str = 'Heat capacity at constant pressure'
        else:
            print('Warning: No. ' + str(i) + ': No property found, continue to next data frame')
            continue
        temperature_index = None
        pressure_index = None
        property_index = None
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

        for data_point in item['data']:  # data_point -> list: 3

            data_point_temperature = data_point[temperature_index][0]
            if pressure_index is None:
                data_point_pressure = 101.325
            else:
                data_point_pressure = data_point[pressure_index][0]
            data_point_value = data_point[property_index][0]
            if not 99. < float(data_point_pressure) < 105.:
                continue
            # if data_point_property == 'heat_capacity' and data_point_property >
            exec('data_point' + str(iter_) + '= DataPoint(data_point_ionic_liquid,'
                                             ' data_point_temperature,'
                                             ' data_point_pressure,'
                                             ' data_point_property,'
                                             ' data_point_value)')
            iter_ += 1

    with open('Data/data_points_list.data', 'wb') as f_data_points_list:
        pickle.dump(DataPoint.list_of_all_Data_Points, f_data_points_list)
    print('Finish: data points list saved to Data/data_points_list.data')
    return DataPoint.list_of_all_Data_Points


if __name__ == '__main__':
    create_data_point()
