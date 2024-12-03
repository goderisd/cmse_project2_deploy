# cmse_project2_deploy

This is just copied over from the full repo that will be turned in on D2L

AI use declaration: Throughout the project i used CHatgpt 4.0 approaximate dates were from 11/20 to 12/3 (while I worked on the project), to aid in the following tasks: Convert AR(p) and VAR(p) code from ICAs and HWs to more compact functions Genral debugging for type and value errors that occured while programming Production of the outline skelton for the pages for the streamlit app Propogate the program features that were developed with the Delta dataset to the United and American Datasets And to comment the final script to ensure ease of read/grade -ing

This Repo houses the full program that can not be deployed due to the size of the files.

Because Github does not allow for >25MB uploads for the CSVs, i have a onedrive link that is here: https://michiganstate-my.sharepoint.com/:f:/g/personal/goderisd_msu_edu/EkGmfvT7RQ5JvsI_nLSY5k0BkeYFIviMVhWDk7wegYhGSA?e=I6PMEx

You can download the csv files and save them on your local machine.

The code is set up to run with the datasets 1 directory above where the python script is housed, if you want to change that please remove the .. on the path for the csv prior to running.

when running from the terminal please ensure that you make a call to increase the server size warning limit to 300MB, otherwise the app will crash.

python -m streamlit run pythonProject2/flightApp_with_storytelling.py --server.maxMessageSize 300

Orginal sources for the data sets:

Source for weather data set: https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events/suggestions?status=pending&yourSuggestions=true

Source for flight delay set: https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023

They juypter file that is in the repo is not required for any execution, it is only there as proof of learning as well as the code required to do the duel merging of the weather and federal holiday datas with teh flight data.
