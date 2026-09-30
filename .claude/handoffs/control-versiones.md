# Handoff: control-versiones (git-devp)
Última actualización: 2026-09-29

## Resumen
Este proyecto no usa el flujo multi-agente estándar (bases de datos → frontend →
backend → calidad → QA): es un proyecto de hardware/ML desarrollado directamente
por el usuario. No existían handoffs previos de otros roles porque no hubo otros
roles involucrados en esta tarea. Mi trabajo fue exclusivamente de publicación
inicial: el repositorio no tenía historial de git en absoluto.

## Qué se hizo
1. Se verificó que la carpeta no era un repositorio git (`git status` -> "not a
   git repository") y que `gh auth status` ya tenía sesión iniciada
   (cuenta `FelipeEsp07`).
2. Se agregó `.gitignore` para Python/MicroPython (`__pycache__/`, `*.pyc`,
   entornos virtuales, cachés de herramientas, archivos de editor/SO).
   Explícitamente NO se excluyeron `*.wav`, `dataset.jsonl`, `graficas/*.png`
   ni `modelo_pesos.py` — son datos y artefactos intencionales del proyecto,
   documentados como tales en el README.
3. `git init`, `git add -A`, commit inicial (`feat: commit inicial del asistente
   de voz local ESP32-S3`) con los 31 archivos existentes (README, código
   MicroPython y de entrenamiento en PC, dataset, audios de ejemplo, gráficas).
4. Rama renombrada a `main`.
5. El servidor MCP de GitHub falló al crear el repositorio (`403 Resource not
   accessible by personal access token`) — el token del MCP no tiene permiso de
   creación de repos. Se usó el CLI `gh` como alternativa, tal como estaba
   autorizado de antemano para este caso:
   `gh repo create asistente-voz-esp32-s3 --public --source=. --remote=origin --push`
6. Se verificó tras el push: remoto `origin` apuntando al repo correcto,
   working tree limpio, visibilidad `PUBLIC` confirmada con
   `gh repo view ... --json visibility`.

## Estado final del repositorio
- Repositorio remoto: https://github.com/FelipeEsp07/asistente-voz-esp32-s3
  (público)
- Rama principal: `main`, con un único commit (`f8d669f`, el commit inicial),
  ya empujado y sincronizado con `origin/main`.
- Working tree limpio, sin cambios pendientes.
- No se identificaron secretos, claves ni credenciales en el contenido
  publicado (confirmado por el usuario y verificado visualmente al revisar
  el listado de archivos).

## Decisiones y supuestos
- No se aplicó el checklist de "cinco handoffs" (bases de datos, frontend,
  backend, calidad, QA) porque no existen esos archivos ni ese flujo en este
  proyecto — es la primera vez que se versiona, y no hubo trabajo de otros
  agentes que fusionar. Si en el futuro este proyecto empieza a usar ese flujo,
  sí debe verificarse antes de cualquier fusión posterior.
- Se usó `gh` CLI en vez del MCP de GitHub porque el MCP devolvió 403 al
  intentar crear el repositorio (permiso insuficiente del token), siguiendo
  la instrucción explícita del usuario de usar `gh` como alternativa en ese
  caso.
- Se creó esta carpeta `.claude/handoffs/` porque no existía; a partir de
  ahora, si el proyecto adopta el flujo multi-agente, este archivo debe
  actualizarse en cada fusión futura.

## Bloqueado por / dependo de
Nada pendiente para esta tarea.

## Pendientes (TODO)
- Ninguno relacionado con control de versiones. El README ya documenta
  pendientes de producto (reconocer más frases, ampliar el dataset).

## Estado de pruebas
No aplica control de versiones per se — no se corrieron tests porque esta
tarea fue exclusivamente de publicación de repositorio, no de cambio de
código. El contenido publicado es el que el usuario ya tenía funcionando
localmente.
