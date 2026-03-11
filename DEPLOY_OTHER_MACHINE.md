# Deploy para Outra Maquina

Este projeto ja consegue rodar em outra maquina, mas voce precisa separar duas coisas:

1. Codigo (vai por `git commit` + `git push`)
2. Pesos treinados (`model_final.pth`, que normalmente nao vai para o git)

## Fluxo recomendado (mais simples)

Na maquina de treino:

```bash
bash scripts/build_inference_bundle.sh
```

Isso gera um pacote em `deploy/` com:
- codigo necessario para inferencia
- configuracoes
- `model_final.pth`

Na outra maquina:

```bash
tar -xzf diffdet4sar_inference_*.tar.gz
cd diffdet4sar_inference_*
bash scripts/setup_runtime_env.sh
source .venv/bin/activate
python infer_tif_dir.py \
  --config-file configs/diffdet.aircraft.single_class.yaml \
  --weights model_final.pth \
  --input-dir /caminho/imagens_tif \
  --output-dir /caminho/saida \
  --recursive
```

## Fluxo alternativo (git + copia manual dos pesos)

1. Suba apenas o codigo no GitHub.
2. Na outra maquina, faça `git clone`.
3. Copie `output_aircraft_single_class/model_final.pth` por `scp/rsync`.
4. Execute setup e inferencia:

```bash
bash scripts/setup_runtime_env.sh
source .venv/bin/activate
python infer_tif_dir.py \
  --config-file configs/diffdet.aircraft.single_class.yaml \
  --weights /caminho/model_final.pth \
  --input-dir /caminho/imagens_tif \
  --output-dir /caminho/saida \
  --recursive
```

## Importante antes do push

Se a pasta `.venv/` apareceu versionada no seu git, remova do indice (sem apagar local):

```bash
git rm -r --cached .venv
git commit -m "Stop tracking local virtual environment"
```

Sem isso, o push fica enorme e dificil de reproduzir em outra maquina.
