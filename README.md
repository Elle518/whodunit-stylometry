# Whodunit Stylometry

Este repositorio contiene el código y los recursos relacionados con el Trabajo de Fin de Grado (TFG) titulado "Modelado estilístico de la novela de misterio - Un análisis computacional de autores canónicos".

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

## Uso de git hook para formatear el código antes de cada commit

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
