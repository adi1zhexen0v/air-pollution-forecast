import os

stations = [
  {
    'folder': os.path.join('data', 'raw', 'station_1'),
    'name': 'Gymnasium 39',
    'lat': 43.348459,
    'lon': 76.84291
  },
  {
    'folder': os.path.join('data', 'raw', 'station_2'),
    'name': 'School 177',
    'lat': 43.357966,
    'lon': 76.921398
  },
  {
    'folder': os.path.join('data', 'raw', 'station_3'),
    'name': 'Elaman 105',
    'lat': 43.36754,
    'lon': 76.945592
  },
  {
    'folder': os.path.join('data', 'raw', 'station_4'),
    'name': 'School 32',
    'lat': 43.372796,
    'lon': 76.991409
  },
  {
    'folder': os.path.join('data', 'raw', 'station_5'),
    'name': 'Iliyski trakt',
    'lat': 43.342767,
    'lon': 76.981528
  },
  {
    'folder': os.path.join('data', 'raw', 'station_6'),
    'name': 'Zorge street, 32',
    'lat': 43.338002,
    'lon': 76.952223
  },
  {
    'folder': os.path.join('data', 'raw', 'station_7'),
    'name': 'Aynabulak 3',
    'lat': 43.326581,
    'lon': 76.920906
  },
  {
    'folder': os.path.join('data', 'raw', 'station_8'),
    'name': 'School 137',
    'lat': 43.317777,
    'lon': 76.911619
  },
  {
    'folder': os.path.join('data', 'raw', 'station_9'),
    'name': 'Kotelnikova St.',
    'lat': 43.310559,
    'lon': 76.94206
  },
  {
    'folder': os.path.join('data', 'raw', 'station_10'),
    'name': 'Jana Kairat',
    'lat': 43.312075,
    'lon': 77.001754
  },
  {
    'folder': os.path.join('data', 'raw', 'station_11'),
    'name': 'Atyrau 54',
    'lat': 43.296422,
    'lon': 76.992052
  },
  {
    'folder': os.path.join('data', 'raw', 'station_12'),
    'name': 'Zhetysu 47',
    'lat': 43.291954,
    'lon': 76.990383
  },
  {
    'folder': os.path.join('data', 'raw', 'station_13'),
    'name': 'Kulager',
    'lat': 43.302194,
    'lon': 76.923765
  },
  {
    'folder': os.path.join('data', 'raw', 'station_14'),
    'name': 'Shamiyeva St.',
    'lat': 43.283152,
    'lon': 76.928097
  },
  {
    'folder': os.path.join('data', 'raw', 'station_15'),
    'name': 'Ryskulova 81',
    'lat': 43.27864,
    'lon': 76.903186
  },
  {
    'folder': os.path.join('data', 'raw', 'station_17'),
    'name': 'Kokkainar Micro',
    'lat': 43.289286,
    'lon': 76.841529
  },
  {
    'folder': os.path.join('data', 'raw', 'station_18'),
    'name': 'CHP-2',
    'lat': 43.291054,
    'lon': 76.800777
  },
  {
    'folder': os.path.join('data', 'raw', 'station_20'),
    'name': 'EkoPost',
    'lat': 43.249577,
    'lon': 76.80691
  },
  {
    'folder': os.path.join('data', 'raw', 'station_21'),
    'name': 'AsiaFood',
    'lat': 43.242868,
    'lon': 76.828894
  },
  {
    'folder': os.path.join('data', 'raw', 'station_22'),
    'name': 'Hospital-7',
    'lat': 43.232526,
    'lon': 76.801056
  },
  {
    'folder': os.path.join('data', 'raw', 'station_23'),
    'name': 'Zhunisova St.',
    'lat': 43.218453,
    'lon': 76.789248
  },
  {
    'folder': os.path.join('data', 'raw', 'station_24'),
    'name': 'DIS-7',
    'lat': 43.206545,
    'lon': 76.776315
  },
  {
    'folder': os.path.join('data', 'raw', 'station_26'),
    'name': 'Abay avenue',
    'lat': 43.225554,
    'lon': 76.861876
  },
  {
    'folder': os.path.join('data', 'raw', 'station_27'),
    'name': 'Mamyr-3',
    'lat': 43.221298,
    'lon': 76.867354
  },
  {
    'folder': os.path.join('data', 'raw', 'station_28'),
    'name': 'Rozybakieva, 270',
    'lat': 43.214958,
    'lon': 76.893317
  },
  {
    'folder': os.path.join('data', 'raw', 'station_30'),
    'name': 'School 192',
    'lat': 43.170202,
    'lon': 76.850401
  },
  {
    'folder': os.path.join('data', 'raw', 'station_31'),
    'name': 'Turuspekova St.',
    'lat': 43.178319,
    'lon': 76.869493
  },
  {
    'folder': os.path.join('data', 'raw', 'station_32'),
    'name': 'Orbita',
    'lat': 43.193682,
    'lon': 76.882578
  },
  {
    'folder': os.path.join('data', 'raw', 'station_33'),
    'name': 'Alatau',
    'lat': 43.176047,
    'lon': 76.896593
  },
  {
    'folder': os.path.join('data', 'raw', 'station_34'),
    'name': 'Nicolas International chain of wineries',
    'lat': 43.194805,
    'lon': 76.910057
  },
  {
    'folder': os.path.join('data', 'raw', 'station_35'),
    'name': 'Kensay',
    'lat': 43.233886,
    'lon': 76.971037
  },
  {
    'folder': os.path.join('data', 'raw', 'station_36'),
    'name': 'KBTU',
    'lat': 43.255326,
    'lon': 76.943861
  },
  {
    'folder': os.path.join('data', 'raw', 'station_37'),
    'name': 'KazNPU',
    'lat': 43.248796,
    'lon': 76.953733
  },
  {
    'folder': os.path.join('data', 'raw', 'station_38'),
    'name': 'Respublika, 4',
    'lat': 43.236886,
    'lon': 76.946489
  },
  {
    'folder': os.path.join('data', 'raw', 'station_39'),
    'name': 'Ritz Palace',
    'lat': 43.227185,
    'lon': 76.959971
  },
  {
    'folder': os.path.join('data', 'raw', 'station_40'),
    'name': 'School 77',
    'lat': 43.21905,
    'lon': 76.949497
  },
  {
    'folder': os.path.join('data', 'raw', 'station_41'),
    'name': 'Tau Samal',
    'lat': 43.206112,
    'lon': 76.982082
  },
]