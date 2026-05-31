# Whodunit Stylometry

Este repositorio contiene el código y los recursos relacionados con el Trabajo de Fin de Grado (TFG) titulado ***"Modelado estilístico de la novela de misterio - Un análisis computacional de autores canónicos"***.

## Creación del entorno virtual

1. Instala en tu máquina `uv` si no lo tienes:

    En MacOS o Linux:

    ```bash
    > curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    En Windows:

    ```powershell
    > powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Crea el entorno virtual a partir del pyproject.toml:

    ```bash
    > uv sync
    ```

3. Comprueba que el paquete `whodunit_stylometry` se ha instalado correctamente:

    ```bash
    > uv run python -c "import whodunit_stylometry; print(whodunit_stylometry.__file__)"
    ```

4. Instala el modelo de spacy:

    ```bash
    > uv run python -m spacy download en_core_web_sm
    ```

## Ejecución de los tests

Para ejecutar la batería de tests con `pytest` desde la raíz del repositorio:

```bash
> uv run pytest -q
```

Para mostrar el nombre y el resultado de cada test:

```bash
> uv run pytest -v
```

## Uso de `git hook` para formatear el código antes de cada commit

Este repositorio utiliza hooks de pre-commit. Es necesario instalar y configurar estos hooks antes de realizar cambios en el repositorio.

Una vez que hayas clonado el repositorio y creado el entorno virtual, ejecuta el siguiente comando para configurar los hooks de pre-commit (solo es necesario hacerlo una vez):

```bash
> uv run pre-commit install
```

Para validar, antes de subir cambios al repositorio, ejecuta:

```bash
> uv run pre-commit run --all-files
```

Esto pasará todos los hooks configurados en los archivos del repositorio.

Si quieres hacerlo solo en un archivo específico, puedes usar:

```bash
> uv run pre-commit run --files ruta/al/archivo
```

Si quieres hacerlo de los archivos modificados en el staging area, puedes usar:

```bash
> uv run pre-commit run
```

En este caso no te hará falta especificar los archivos.

Si pasa los checks, ya puedes hacer commit y push como de costumbre, si no corregirá los errores que haya encontrado y tendrás que volver a subir los cambios al staging area.

Si quieres saltarte los hooks de pre-commit en un commit específico, puedes usar la opción `--no-verify` al hacer el commit:

```bash
git commit --no-verify -m "my_commit"
```

## Para añadir nuevas dependencias al entorno virtual

Cuando queramos añadir nuevas dependencias al entorno virtual, es importante seguir estos pasos para asegurarnos de que el entorno se mantiene actualizado y que los cambios se reflejan correctamente en el control de versiones:

```bash
> uv add <nueva-dependencia>
> uv lock
> uv sync
> git add pyproject.toml uv.lock
> git commit -m "Add <nueva-dependencia>"
> git push
```

## App interactiva

La interfaz de Streamlit permite ejecutar diferentes métodos de análisis estilométrico sobre un corpus organizado como `corpus/autor/*.txt`.

Para ejecutar la app, asegúrate de tener el entorno virtual activo y luego ejecuta:

```bash
> uv run streamlit run app.py
```

## Notebooks experimentales

El directorio `notebooks/` contiene los cuadernos utilizados durante la fase experimental del proyecto. Estos notebooks documentan el proceso de exploración, entrenamiento, evaluación e interpretación de los distintos enfoques aplicados en la memoria.

| Notebook | Objetivo | Entradas principales | Salidas o resultados |
| --- | --- | --- | --- |
| `01_eda_corpus.ipynb` | Realizar el análisis exploratorio inicial del corpus. | Corpus limpio y metadatos. | Estadísticas descriptivas, análisis por autor, métricas de calidad, longitud, riqueza léxica, puntuación y visualizaciones exploratorias. |
| `02_classic_stylometric_tests.ipynb` | Aplicar métodos clásicos de estilometría. | Corpus por autor y obras reservadas para prueba. | Resultados de Mendenhall, Kilgariff y Burrows, distancias por autor y análisis de atribución. |
| `03_supervised_ml_methods.ipynb` | Entrenar y comparar modelos supervisados de atribución. | Rasgos estilométricos, MFW y etiquetas de autor. | Métricas de clasificación, matrices de confusión y comparación entre modelos. |
| `04_mfw_robustness_experiment.ipynb` | Evaluar la robustez del número de palabras funcionales. | Rasgos MFW con distintas configuraciones. | Resultados por número de rasgos, semillas aleatorias y estabilidad del rendimiento. |
| `05_xgboost_base_features_experiment.ipynb` | Probar XGBoost sobre rasgos estilométricos base. | Métricas estilométricas agregadas. | Evaluación adicional de un modelo basado en *boosting*. |
| `06_unsupervised_ml_methods.ipynb` | Analizar agrupamientos no supervisados. | Rasgos estilométricos, MFW y etiquetas solo para evaluación. | Resultados de clustering, ARI, NMI, homogeneidad, completitud, *V-measure*, *silhouette* y visualizaciones. |
| `07_openai_embeddings_models.ipynb` | Generar y analizar *embeddings*. | Textos fragmentados y API de OpenAI. | *Embeddings* por fragmento y obra, similitudes, proyecciones y redes de proximidad. |
| `08_supervised_ml_explainability_lr_coef.ipynb` | Interpretar modelos lineales mediante coeficientes. | Modelo supervisado entrenado y rasgos MFW. | Rasgos más asociados a cada autor y análisis global de coeficientes. |
| `09_supervised_ml_explainability_shap.ipynb` | Aplicar SHAP a modelos supervisados. | Modelo entrenado, datos de test y rasgos del modelo. | Explicaciones globales y locales de predicciones concretas. |
| `10_train_authorship_transformer_colab.ipynb` | Ajustar RoBERTa para atribución de autoría. | Fragmentos de texto etiquetados por autor. | Modelo RoBERTa ajustado, métricas por fragmento y evaluación agregada por documento. |
| `11_transformer_explainability_captum.ipynb` | Analizar atribuciones del *transformer* con Captum. | Modelo RoBERTa ajustado y fragmentos de texto. | Atribuciones a nivel de token para predicciones concretas. |
| `12_transformer_explainability_bertviz.ipynb` | Visualizar atención del *transformer* con BertViz. | Modelo RoBERTa ajustado y ejemplos de texto. | Visualizaciones de atención por capas y cabezas del modelo. |
