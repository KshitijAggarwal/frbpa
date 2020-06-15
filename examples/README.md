# Example Notebooks

Data Format
---
The data should be a dictionary saved in JSON format. 
The keys of the dictionary should be `obs_duration`, `obs_startmjds` and `bursts`. 
Each of which should be a dictionary with keys as names of the different 
telescopes/configurations. The observation durations should be in seconds.  

Example JSON: 
```json
{
    "bursts": {
        "CHIME": [
            50000.1234,
            52000.1234
        ],
        "VLA": [
            53000.1234
        ]
    },
    "obs_duration": {
        "CHIME": [
            720,
            720,
            720,
            720
        ],
        "VLA": [
            3600,
            3600
        ]
    },
    "obs_startmjds": {
        "CHIME": [
            50001.2345,
            50002.2345,
            50003.2345,
            50004.2345
        ],
        "VLA": [
            53000.3456,
            53001.4567
        ]
    }
}
```

Note: `obs_durations` are required to use `pr3_search`. 

Notebook data sources:
---

* r1: Uses FRB121102 burst data from [Rajwade et al (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.tmp.1508R/abstract)
* r2: Uses FRB180814 burst data from [CHIME-FRB](https://www.chime-frb.ca/)
* r3: Uses all published burst data for FRB180916
* r3all: Uses all published burst data for FRB180916 and new bursts from [CHIME-FRB](https://www.chime-frb.ca/)
* 190303: Uses FRB190303 burst data from [CHIME-FRB](https://www.chime-frb.ca/)
