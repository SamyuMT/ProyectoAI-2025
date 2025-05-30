# Proyecto de Análisis de Datos

## Descripción General
Este proyecto analiza datos del DANE (Departamento Administrativo Nacional de Estadística) utilizando Python. Incluye varios notebooks de Jupyter y una librería personalizada para el procesamiento y visualización de datos.

## Tecnologías Utilizadas
- Python 3.13
- Jupyter Notebooks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Librería personalizada (`libreria.py`)

## Fuente de Datos
Los conjuntos de datos deben descargarse desde:
https://microdatos.dane.gov.co/index.php/catalog/852/data-dictionary

Debes descargar los datos correspondientes al segundo, tercer y cuarto trimestre y ubicarlos en el directorio `data/raw` y crear un subcarperta `data/processed`.

## Flujo de Trabajo
Para analizar los datos correctamente, sigue estos pasos en orden:

1. **Limpieza de Datos**: Ejecuta primero `limpieza.ipynb` para limpiar y preparar los datos.
2. **Análisis Inicial**: Ejecuta `pruebaI.ipynb` para la primera etapa de análisis.
3. **Análisis Secundario**: Continúa con `pruebaII.ipynb`.
4. **Análisis Completo**: Finalmente, ejecuta los notebooks que comienzan con el prefijo "full".

## Librería Personalizada
El proyecto utiliza una librería personalizada (`libreria.py`) que contiene funciones diseñadas específicamente para este flujo de procesamiento de datos.

## Instrucciones de Instalación
1. Clona este repositorio.
2. Instala las dependencias: `pip install -r requirements.txt`
3. Descarga los conjuntos de datos requeridos y colócalos en `data/raw/`
4. Sigue el flujo de trabajo en el orden descrito arriba.