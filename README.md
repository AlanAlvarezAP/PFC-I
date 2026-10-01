# PFC I
Implementación y reproducción de modelos de Graph Neural Networks para predicción epidemiológica con motivos de comparación academica para el curso de PFC-I

**Información del Repositorio**
- **Alumno**: Alan Alvarez
- **Curso**: Proyecto Final de la Carrera I
- **Semestre**: Ccomp 8-1 / 2026-2

## Modelos y Agradecimientos
Los modelos que se implementaron serán los siguientes, se agradece con los respectivos creditos:

* **[MepoGNN]** — *Metapopulation Epidemic Forecasting with Graph Neural Networks*
  - **Autores:** Qi Cao, Renhe Jiang, Chuang Yang, Zipei Fan, Xuan Song y Ryosuke Shibasaki
* **[BDSTGNN]** — **Backbone-based Dynamic Spatio-Temporal GraphNeuralNetwork for epidemic forecasting**  
  - **Autores:** Junkai Mao, Yuexing Han, Gouhei Tanaka y Bing Wang

## Estructura

- `MepoGNN/`: implementación del modelo MepoGNN
- `BDSTGNN/`: implementación del modelo BDSTGNN

## Requisitos

- Python 3.11
- Git
- Docker (futuro)

**Nota**: Hay otros requerimientos en los modelos cada uno se puede ver en su respectivo requirements.txt

## Estado del proyecto

En desarrollo

## Instalación y Compilación
Para la instalación y compilación respectiva del repositorio y ejecución de los modelos:

**Instalación manual**
1. Clonar Repositorio: En el repositorio copiar el enlace y clonarlo con **https://github.com/AlanAlvarezAP/PFC-I.git**
2. Instalación de Python: Tener instalado la versión de Python 3.11 en caso contrario realizar en la terminal dentro de la carpeta **py install 3.11**
3. Creación del ambiente: Ingresar a la carpeta del modelo a querer usar y crear su .env con **py -3.11 -m venv .venv**
4. Instalación de requerimientos: Para poder correr el codigo instalar las dependencias con **pip install -r requirements.txt**
5. Correr el codigo: Para empezar a ejecutar el codigo usar el comando **python ./main.py**

Disfruta :D!!!

**Con Docker**
*Pronto...*