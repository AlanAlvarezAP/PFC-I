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

### Ejecutar la comparación con Docker

Abre Docker Desktop y una terminal en la raíz del repositorio, donde se encuentra `compose.yaml`. Las carpetas `BDSTGNN/data/` y `MepoGNN/data/` deben contener los datasets del proyecto.

**Con GPU NVIDIA**, construye la imagen y abre el menú:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm experimento
```

**Con CPU**, utiliza estos comandos:

```powershell
docker compose build
docker compose run --rm experimento
```

`build` prepara el entorno. Solo necesitas repetirlo cuando cambies el código incluido en la imagen, el `Dockerfile` o las dependencias. `run` ejecuta `comparar_activos.py` y abre el siguiente menú:

| Opción | Acción |
|---|---|
| 1 | Entrena ambos modelos desde cero, guarda sus pesos y evalúa. |
| 2 | Carga los pesos guardados y evalúa ambos modelos. |
| 3 | Entrena ambos modelos desde cero y guarda sus pesos. |
| 4 | Revisa las fechas, regiones y ventanas disponibles. |
| 0 | Sale del programa. |

En la primera ejecución, elige **1** para generar los pesos y obtener las métricas. En las siguientes, elige **2** para repetir la evaluación sin entrenar. Las opciones **1** y **3** reemplazan los pesos anteriores.

Los checkpoints se guardan en `pesos/` y las predicciones y métricas en `resultados_comparacion/`. Ambas carpetas permanecen en el equipo al cerrar el contenedor.

La comparación externa evalúa del **1 de julio al 5 de septiembre de 2022**, con 14 días de historia y 14 de predicción. Compara los activos predichos directamente por BDSTGNN con los activos derivados de MepoGNN Adaptive. Para reproducir las pruebas originales de cada paper, ejecuta el `main.py` de la carpeta del modelo siguiendo la instalación manual.

### Archivos ignorados

- `.gitignore` excluye del control de versiones los entornos virtuales, la caché de Python, los pesos y los resultados generados.
- `.dockerignore` limita los archivos que se envían al construir la imagen. Al ejecutar el programa, Docker lee los datasets desde las carpetas locales montadas por Compose.
