import os
import os.path
import shutil


def collect_cosmo():
    for _, sub_dir, __ in os.walk('Cosmo_file/a_xyz2cosmo'):
        for dir_name in sub_dir:
            try:
                shutil.copy('Cosmo_file/a_xyz2cosmo/' + dir_name + '/' + dir_name + '.cosmo', 'Cosmo_file/anion_cosmo')
            except FileNotFoundError:
                print(dir_name)

    for _, sub_dir, __ in os.walk('Cosmo_file/c_xyz2cosmo'):
        for dir_name in sub_dir:
            try:
                shutil.copy('Cosmo_file/c_xyz2cosmo/' + dir_name + '/' + dir_name + '.cosmo', 'Cosmo_file/cation_cosmo')
            except FileNotFoundError:
                print(dir_name)


def create_input_file(temperature, min_, max_):
    input_string = ''
    no = 0
    for i in range(min_, max_ + 1):
        input_string += '<DbuCompoundCombination anion="anion_' + str(temperature) + '_' + str(i) +\
                            '.cosmo" cation="cation_' + str(temperature) + '_' + str(i) +\
                            '.cosmo">' + str(no) + '</DbuCompoundCombination>\n'
        no += 1
    return input_string
