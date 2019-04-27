from netconstructor.datareader import load_from_csv, load_from_img
from test import TEST_ROOT_DIR


def test_load_from_csv_3_4():
    data = load_from_csv(f'{TEST_ROOT_DIR}/resource/data_3_4.csv')

    assert data.shape == (3, 4)


def test_load_from_csv_2_2():
    data = load_from_csv(f'{TEST_ROOT_DIR}/resource/data_2_2.csv', ';')

    assert data.shape == (2, 2)


def test_load_from_csv_1_1():
    data = load_from_csv(f'{TEST_ROOT_DIR}/resource/data_1_1.txt')

    assert data.shape == ()  # TODO: fix shape for corner case


def test_load_from_img():
    data = load_from_img(f'{TEST_ROOT_DIR}/resource/img/5.png')

    assert len(data) == 1


def test_load_from_img_dir():
    data = load_from_img(f'{TEST_ROOT_DIR}/resource/img')

    assert len(data) == 2


def test_load_from_img_dir_rec():
    data = load_from_img(f'{TEST_ROOT_DIR}/resource/img', rec=True)

    assert len(data) == 3
