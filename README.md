# Miniproyecto1
Extracción de características
Creado por Jhonatan Sossa y Juan Pablo Quinchía

En el respositorio se encuentran dos documentos .ipynb, además de los datos usados
emotions_fuctions.ipynb y emotions.ipynb
En el primero se encuentran todas las funciones utilizadas para la extracción de las características y en el segundo se encuenta el main. 
Para la una correcta ejecución, es necesario crear una carpeta llamada Miniproyecto1 en google drive y poner allí ambos documentos y los datos. En caso de que no se cree la carpeta llamada Miniproyecto1, entonces se debe cambiar la ruta especificada en la línea %cd '/content/drive/My Drive/Miniproyecto1' por la nueva ruta. 
Sólo es necesario correr el archivo emotions.ipynb. La ejecución tarda algunos minutos, pues en la extracción de características se usaron ventanas de 0.02 segundos con overlap del 50%, lo que hace el proceso un poco más lento.
En caso de que no se desee hacer toda la ejecución, también se sube al repositorio la matriz de características (feat_mat.p), el vector Arousal (Arousal.p) y el vector Valence (Valence.p). Se pueden cargar estos tres archivos descomentando la celda 9 y se terminando de ejecutar las celdas siguientes. (Sigue siendo necesario ejecutar las primeras 7 celdas)
El archivo feats.pkl contiene la matriz de características incluyendo el vector de etiquetas en un dato de tipo pandas.