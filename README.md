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
- `comparar_activos.py`: comparación BDSTGNN vs MepoGNN (entrena, evalúa y exporta CSV)
- `datos_y_metricas.py`, `metrics.py`: carga de datos externos y métricas
- `Dockerfile`, `compose.yaml`, `compose.gpu.yaml`, `requirements-docker.txt`, `.dockerignore`: ejecución de la comparación en Docker
- `graficar_comparacion.py`: consolida los CSV y genera las gráficas

## Requisitos

- Python 3.11
- Git
- Docker Desktop (opcional, para ejecutar la comparación en contenedor)

**Nota**: Hay otros requerimientos en los modelos cada uno se puede ver en su respectivo requirements.txt

## Estado del proyecto

En desarrollo, version inicial terminada (replicación y experimentos) v1.0

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

Abre Docker Desktop y luego PowerShell en la raíz del proyecto:

```powershell
cd D:\PFC-I
```

**Con GPU NVIDIA (RTX 2060)**

La primera vez, construye la imagen y abre el menú:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm experimento
```

El contenedor ejecuta `comparar_activos.py`, así que todo lo que escribas
después de `experimento` se pasa como argumento. Sin argumentos abre un menú
(elige **1** para entrenar y evaluar BDSTGNN y MepoGNN, **5** para entrenar
MepoGNN con activos y **6** para evaluar los tres). Los pesos se guardan en
`pesos/` y las métricas en `resultados_comparacion/`.

También puedes ejecutar cada paso directamente:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm experimento --accion entrenar-evaluar --protocolo comun
docker compose -f compose.yaml -f compose.gpu.yaml run --rm experimento --accion entrenar-activos --protocolo comun
docker compose -f compose.yaml -f compose.gpu.yaml run --rm experimento --accion evaluar-tres --protocolo comun
```

**Con CPU**

```powershell
docker compose build
docker compose run --rm experimento
docker compose run --rm experimento --accion evaluar-tres --protocolo comun
```

Solo necesitas repetir `build` si cambias el código incluido en la imagen
o sus dependencias.

**Nota**: si cambias de protocolo (ver abajo) o modificas el entrenamiento,
borra `pesos/` y `resultados_comparacion/` y vuelve a entrenar. Los pesos
guardados con otro protocolo no son comparables.

### Qué experimento ejecuta cada comando

Docker abre `comparar_activos.py`. Este script compara BDSTGNN y MepoGNN
sobre un test externo del **1 de julio al 5 de septiembre de 2022**
(54 ventanas), con 14 días de historia y 14 días de predicción
(horizontes de 1 a 14 días).

**Protocolo común (por defecto, `--protocolo comun`)**

Para comparar únicamente la arquitectura, ambos modelos entrenan con el mismo
periodo, **2022-01-15 a 2022-06-14**, con el mismo reparto
(entrenamiento `[:-60]`, validación `[-60:-30]`) y ventanas de 14 días con
paso 1 (64 ventanas de entrenamiento y 3 de validación). Además comparten
200 épocas máximas y early stopping con paciencia 15 restaurando el mejor
checkpoint. Cada modelo conserva su semilla (BDSTGNN 1234, MepoGNN 3) y su
learning rate original.

Con `--protocolo paper` cada modelo entrena con sus datos originales
(BDSTGNN: 2022-01-15 a 2022-06-14; MepoGNN: 2020-04-01 a 2021-09-21).

**Modelos comparados en casos activos**

- **BDSTGNN**: predice directamente el nivel de casos activos.
- **MepoGNN derivado**: entrenado con casos nuevos; los activos salen de su dinámica SIR.
- **MepoGNN corregido**: variante propia entrenada con activos como objetivo (no es la versión del paper).

**Pasos del experimento** (se eligen en el menú o con `--accion`)

```powershell
--accion entrenar-evaluar   # BDSTGNN y MepoGNN (casos nuevos)
--accion entrenar-activos   # MepoGNN corregido
--accion evaluar-tres       # evalua los tres con los pesos guardados
```

**Gráficas y CSV consolidados**

```powershell
python graficar_comparacion.py
```

Con Docker (el entrypoint de la imagen es `comparar_activos.py`, por eso se
sobrescribe):

```powershell
docker compose run --rm --entrypoint python experimento graficar_comparacion.py
```

Genera `comparacion_metricas.csv`, `comparacion_diaria_ancha.csv`,
`comparacion_diaria_nacional.csv` y la carpeta `graficas/` (series de tiempo
nacionales y por prefectura, barras de métricas, error por horizonte y MAE
por prefectura).

### Reproducir los experimentos de los papers

Para reproducir los experimentos individuales de los papers, activa tu
entorno virtual y ejecuta el `main.py` del modelo correspondiente.

**BDSTGNN** — historia de 5 días y horizontes de 5, 10, 15 y 20 días:

```powershell
cd D:\PFC-I\BDSTGNN
python main.py
```

**MepoGNN** — elige la variante que quieras entrenar y evaluar:

```powershell
cd D:\PFC-I\MepoGNN
python main.py --graph Adaptive --seed 3 --no_plot
```

Para la variante dinámica, desde esa misma carpeta:

```powershell
python main.py --graph Dynamic --seed 3 --no_plot
```