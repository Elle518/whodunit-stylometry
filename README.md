# Whodunit Stylometry

Este repositorio contiene el código y los recursos relacionados con mi Trabajo de Fin de Grado (TFG) titulado "Modelado estilístico de la novela de misterio - Un análisis computacional de autores canónicos".

## Creación del entorno virtual

```bash
> poetry init
> poetry env use python3.11
> poetry install
> poetry add pandas numpy
> poetry add --group dev pytest ruff black
> poetry add --group dev ipykernel ipywidgets
```

Activa el entorno virtual:

```bash
> source .venv/bin/activate
```

Instalar el src como paquete editable:

```bash
> pip install -e .
```

## Uso de git hook para formatear el código antes de cada commit

Este repositorio utiliza hooks de pre-commit. Es necesario instalar y configurar estos hooks antes de realizar cambios en el repositorio.

Una vez que hayas clonado el repositorio, ejecuta el siguiente comando para configurar los hooks de pre-commit (solo es necesario hacerlo una vez):

```bash
> poetry add -G dev pre-commit
> poetry run pre-commit install
```

Para validar, antes de subir cambios al repositorio, ejecuta:

```bash
> poetry run pre-commit run --all-files
```

Esto pasará todos los hooks configurados en los archivos del repositorio.

Si quieres hacerlo solo en un archivo específico, puedes usar:

```bash
> poetry run pre-commit run --files ruta/al/archivo
```

Si quieres hacerlo de los archivos modificados en el staging area, puedes usar:

```bash
> poetry run pre-commit run
```

En este caso no te hará falta especificar los archivos.

Si pasa los checks, ya puedes hacer commit y push como de costumbre, si no corregirá los errores que haya encontrado y tendrás que volver a subir los cambios al staging area.

Si quieres saltarte los hooks de pre-commit en un commit específico, puedes usar la opción `--no-verify` al hacer el commit:

```bash
git commit --no-verify -m "my_commit"
```
