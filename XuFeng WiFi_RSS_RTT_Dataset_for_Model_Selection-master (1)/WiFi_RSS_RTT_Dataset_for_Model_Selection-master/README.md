WiFi dataset for model selection
========================
We provide three real-world datasets for indoor positioning model selection purpose. We divided the area of interest was divided into discrete grids and labelled them with correct ground truth coordinates and the LoS APs from the grid. The dataset contains WiFi RTT and RSS signal measures and is well separated so that training points and testing points will not overlap. Please find the datasets in the 'data' folder.

1. Lecture theatre
This is a entirely LOS scenario with 5 APs. 60 scans of WiFi RTT and RSS signal measures were collected at each reference point (RP).
Below is the layout of the testbed. The orange dots indicate the locations of the RTT-enabled Access Point (AP). All measurements are taken in the grey area.
![201](https://github.com/Fx386483710/WiFi-dataset/assets/101070586/3cfca446-c1e2-49ac-bbbe-3cd42f19c0b1)

2. Corridor
This is a entirely NLOS scenario with 4 APs. 60 scans of WiFi RTT and RSS signal measures were collected at each reference point (RP).
Below is the layout of the testbed. The orange dots indicate the locations of the RTT-enabled Access Point (AP). All measurements are taken in the grey area.
![cor_4th](https://github.com/Fx386483710/WiFi-dataset/assets/101070586/e4bc6c31-ef88-41b4-b7d0-42540b3fd0ad)

3. Office
This is a mixed LOS-NLOS scenario with 5 APs. At least one AP was NLOS for each RP. 60 scans of WiFi RTT and RSS signal measures were collected at each reference point (RP).
Below is the layout of the testbed. The orange dots indicate the locations of the RTT-enabled Access Point (AP). All measurements are taken in the grey area.
![405](https://github.com/Fx386483710/WiFi-dataset/assets/101070586/e87b31ed-7677-4c52-ad7c-c882c71059d3)

The features of the dataset
========================

The features of the lecture theatre dataset are as follows:

```
Testbed area:	15 × 14.5 m2
Grid size: 0.6 × 0.6 m2
Number of reference points: 120
Samples per reference point: 60
Number of all data samples: 7,200
Number of training samples: 5,400
Number of testing samples: 1,800
Signal measure: WiFi RTT, WiFi RSS
Note: Entirely LOS
```

The features of the corricor dataset are as follows:

```
Testbed area:	35 × 6 m2
Grid size: 0.6 × 0.6 m2
Number of reference points: 114
Samples per reference point: 60
Number of all data samples: 6,840
Number of training samples: 5,130
Number of testing samples: 1,710
Signal measure: WiFi RTT, WiFi RSS
Note: Miexed LOS-NLOS. At least one AP was NLOS for each RP.
```

The features of the office dataset are as follows:

```
Testbed area:	18 × 5.5 m2
Grid size: 0.6 × 0.6 m2
Number of reference points: 108
Samples per reference point: 60
Number of all data samples: 6,480
Number of training samples: 4,860
Number of testing samples: 1,620
Signal measure: WiFi RTT, WiFi RSS
Note: Entirely NLOS
```

Dataset explanation
========================

The columns of the dataset are as follows:

```
Column 'X': the X coordinates of the sample.
Column 'Y': the Y coordinates of the sample.
Column 'AP1 RTT(mm)', 'AP2 RTT(mm)', ..., 'AP5 RTT(mm)': the RTT measure from corresponding AP at a reference point.
Column 'AP1 RSS(dBm)', 'AP2 RSS(dBm)', ..., 'AP5 RSS(dBm)': the RSS measure from corresponding AP at a reference point.
Column 'LOS APs': indicating which AP has a LOS to this reference point.
```

Please note:

* The RSS value -200 dBm indicates that the AP is too far away from the current reference point and no signals could be heard from it. 

* The RTT value 100,000 mm indicates that no signal is received from the specific AP.
